import cv2
import numpy as np
from scipy.spatial import distance
import statistics
from keras.models import load_model

np.set_printoptions(precision=6, suppress=True)

class Grabber:
    def __init__(self, image, model):
        self.img = image
        self.height = int(self.img.shape[0])
        self.width = int(self.img.shape[1])
        self.nn_model = model
        self.img_to_paint_on = self.img.copy()

    def pre_process(self):
        '''
        Converts image to grayscale, smoothens the image by applying bilateral filter,
        applies Canny edge detector
        :return: image with applied tranformations, making it easier for further processing
        '''
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)

        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        return edges

    def find_biggest_contour(self, image):
        contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) != 0:
            areas = list(map(cv2.contourArea, contours))
            biggest_contour_index = np.argmax(areas)
            biggest_contour = np.squeeze(contours[biggest_contour_index])
        else:
            biggest_contour = 0
        return biggest_contour

    def contours_processing(self, image):
        '''
        Finds all contours in the image, choose the one with the biggest area, and extract from it
        4 corners, which are going to be used to indicate our Region Of Interest (the sudoku board)
        :param image: Image in which we are searching for contours
        :return: A tuple of 4 corners of the sudoku board found in the image
        '''
        biggest_contour = self.find_biggest_contour(image)
        try:
            if biggest_contour == 0:
                biggest_contour = np.array([0,0])

        except Exception as err:
            pass

        if len(biggest_contour.shape) >= 2:
            cv2.drawContours(self.img_to_paint_on, [biggest_contour], -1, (0, 255, 0), 2)

            biggest_contour_x = biggest_contour[:, 0]
            biggest_contour_y = biggest_contour[:, 1]

            medium_x = (max(biggest_contour_x) - min(biggest_contour_x)) / 2 + min(biggest_contour_x)
            medium_y = (max(biggest_contour_y) - min(biggest_contour_y)) / 2 + min(biggest_contour_y)

            top_left = []
            top_right = []
            bottom_left = []
            bottom_right = []

            for i in biggest_contour:
                if i[0] <= medium_x:  # left
                    if i[1] <= medium_y:
                        top_left.append((i[0], i[1]))
                    else:
                        bottom_left.append((i[0], i[1]))
                else:  # right
                    if i[1] <= medium_y:
                        top_right.append((i[0], i[1]))
                    else:
                        bottom_right.append((i[0], i[1]))

            try:
                top_left_ROI = top_left[np.argmin(np.sum(top_left, axis=1))]
                bottom_right_ROI = bottom_right[np.argmax(np.sum(bottom_right, axis=1))]
                bottom_left_ROI = bottom_left[np.argmax(list(map(sum, bottom_left * np.array([-1, 1]))))]
                top_right_ROI = top_right[np.argmax(list(map(sum, top_right * np.array([1, -1]))))]
                return (top_left_ROI, top_right_ROI, bottom_right_ROI, bottom_left_ROI), biggest_contour
            except:
                return ((0, 10), (0, 10), (0, 10), (0, 10)), np.array([0,0])

        else:
            print('No contour was found')
            return ((0, 10), (0, 10), (0, 10), (0, 10)), np.array([0,0])

    def markROI(self, corners):
        for corner_coords in corners:
            cv2.circle(self.img_to_paint_on, corner_coords, 5, (0, 0, 255), -1)

    def warp_image(self, corners):
        width_bottom = distance.euclidean(corners[3], corners[2]) #BL, BR
        width_top = distance.euclidean(corners[0], corners[1])    #TL, TR
        height_left = distance.euclidean(corners[3], corners[0])  #BL, TL
        height_right = distance.euclidean(corners[2], corners[1]) #BR, TR

        maxWidth = int(max(width_bottom, width_top))
        maxHeight = int(max(height_left, height_right))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        self.trans_matrix = cv2.getPerspectiveTransform(np.float32(corners), dst)
        self.img_warped = cv2.warpPerspective(self.img, self.trans_matrix, (maxWidth, maxHeight))

    def eliminate_grid_lines(self):
        inverted = cv2.bitwise_not(self.img_warped)
        gray = cv2.cvtColor(inverted, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 15, 11, 11)
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 45, -2)
        horizontal = np.copy(bw)
        vertical = np.copy(bw)
        cols = horizontal.shape[1]
        horizontal_size = cols // 10
        try:
            horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
            horizontal = cv2.erode(horizontal, horizontalStructure)
            horizontal = cv2.dilate(horizontal, horizontalStructure)
        except Exception as err:
            print(err)
        rows = vertical.shape[0]
        verticalsize = rows // 10
        try:
            verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
            vertical = cv2.erode(vertical, verticalStructure)
            vertical = cv2.dilate(vertical, verticalStructure)
        except Exception as err:
            print(err)
        with_grid_extracted = bw - (vertical + horizontal)
        kernel = np.ones((2, 2), np.uint8)
        with_grid_extracted = cv2.dilate(with_grid_extracted,kernel, iterations=1)
        kernel = np.ones((3, 3), np.uint8)
        with_grid_extracted = cv2.erode(with_grid_extracted,kernel, iterations=1)
        return with_grid_extracted

    def find_digits(self, warped_grid_extracted):
        height_step = int(warped_grid_extracted.shape[0] / 9)
        width_step = int(warped_grid_extracted.shape[1] / 9)
        no = 0
        food_for_nn = []
        is_this_square_a_digit = np.ones((9, 9))
        more_likely_to_be_1 = []
        heights = []
        ratios = []

        for i in range(9):
            for j in range(9):
                bounding_rect = warped_grid_extracted[height_step * j: height_step * (1 + j),
                                width_step * i: width_step * (1 + i)]
                biggest_contour = self.find_biggest_contour(bounding_rect)
                biggest_contour = biggest_contour + np.array([width_step*i, height_step*j])

                if len(biggest_contour.shape) >= 2:
                    boundary_left = min(biggest_contour[:, 0])
                    boundary_right = max(biggest_contour[:, 0])
                    boundary_top = min(biggest_contour[:, 1])
                    boundary_bottom = max(biggest_contour[:, 1])

                    cX = int((boundary_right + boundary_left)/2)
                    cY = int((boundary_bottom + boundary_top)/2)

                    rect_height = int(boundary_bottom - boundary_top)
                    rect_width = int(boundary_right - boundary_left)

                    heights.append(rect_height)

                    if rect_width <= 0:
                        ratio = 0
                    else:
                        ratio = rect_height/rect_width

                    ratios.append(ratio)

                    if ratio < 1 or ratio > 5:
                        # If contour is too wide or too tall - it's probably not a digit
                        is_this_square_a_digit[j, i] = -1
                        square_for_nn = np.zeros((2, 2))

                    else:
                        margin = rect_height // 2 + 5
                        if margin < 4:
                            margin = 7

                        if cX - margin < 0:
                            if cY - margin < 0:
                                square_for_nn = warped_grid_extracted[0:cY + margin,
                                               0:cX + margin]
                            else:
                                square_for_nn = warped_grid_extracted[cY - margin:cY + margin,
                                               0:cX + margin]
                        elif cY - margin < 0:
                            square_for_nn = warped_grid_extracted[0:cY + margin,
                                           cX - margin:cX + margin]
                        else:
                            square_for_nn = warped_grid_extracted[cY - margin:cY + margin,
                                           cX - margin:cX + margin]
                else:
                    heights.append(0)
                    ratios.append(0)
                    is_this_square_a_digit[j, i] = -1
                    square_for_nn = np.zeros((2,2))
                no += 1
                food_for_nn.append(square_for_nn)
        try:
            height_median = statistics.median([i for i in heights if i > 3])
            ratios_median = statistics.median([ratios[i] for i in range(81) if heights[i] > 3])
            for i, hei in enumerate(heights):
                if hei < height_median * 0.85:
                    is_this_square_a_digit[i%9, i//9] = -1

            for i, rat in enumerate(ratios):
                if rat > ratios_median * 1.15:
                    more_likely_to_be_1.append((i%9, i//9))
        except Exception as err:
            pass

        return food_for_nn, is_this_square_a_digit, more_likely_to_be_1

    def recognize_digits(self, digit_images, is_a_digit, more_likely_1):
        output_sudoku_board = -1 * np.ones((9, 9))
        for i, d_image in enumerate(digit_images):
            if is_a_digit[i%9, i//9] < 0:
                output_sudoku_board[i%9, i//9] = 0
            else:
                if d_image.shape[0] != 28 or d_image.shape[1] != 28:
                    d_image = cv2.resize(d_image, (28, 28), interpolation=cv2.INTER_AREA)
                d_image = d_image.reshape(d_image.shape[0], d_image.shape[1], 1)
                input_eval = d_image.astype('float32')
                input_eval /= 255
                input_eval = np.expand_dims(input_eval, axis=0)
                prob = self.nn_model.predict(input_eval)
                predicted_class = np.argmax(prob[0]) + 1
                if predicted_class == 1:
                    if (i%9, i//9) not in more_likely_1:
                        prob[0][0] -= 0.1
                        prob[0][6] += 0.1
                        predicted_class = np.argmax(prob[0]) + 1
                if predicted_class == 4:
                    if (i % 9, i // 9) in more_likely_1:
                        prob[0][0] += 0.5
                        prob[0][3] -= 0.5
                        predicted_class = np.argmax(prob[0]) + 1
                if predicted_class == 7:
                    if (i % 9, i // 9) in more_likely_1:
                        prob[0][0] += 0.2
                        prob[0][6] -= 0.2
                        predicted_class = np.argmax(prob[0]) + 1
                if max(prob[0]) < 0.4:
                    output_sudoku_board[i % 9, i // 9] = 0
                else:
                     output_sudoku_board[i % 9, i // 9] = predicted_class
        return output_sudoku_board.astype(int)


class Solver:
    def __init__(self, board):
        self.board_to_solve = board

    def check_if_board_valid(self):
        # Method used for validating the recognition process -
        # if the recognized board is not valid, not all digits were recognized corectly
        for row in self.board_to_solve:
            elements_in_row = set()
            for element in row:
                if element != 0:
                    if element not in elements_in_row:
                        elements_in_row.add(element)
                    else:
                        return False
        for col in self.board_to_solve.T:
            elements_in_col = set()
            for element in col:
                if element != 0:
                    if element not in elements_in_col:
                        elements_in_col.add(element)
                    else:
                        return False
        for i in range(3):
            for j in range(3):
                small_square = self.board_to_solve[3 * j: 3 * (j + 1), 3 * i: 3 * (i + 1)]
                square_flat = small_square.flatten()
                elements_in_square = set()
                for element in square_flat:
                    if element != 0:
                        if element not in elements_in_square:
                            elements_in_square.add(element)
                        else:
                            return False
        return True

    def find_next_empty_cell(self):
        # Returns x and y coordinates of next empty cell
        for i in range(9):
            for j in range(9):
                if self.board_to_solve[i][j] == 0:
                    row = i
                    col = j
                    return row, col
        return None

    def which_square(self, row, col):
        # Returns an offset suggesting in which square we are
        # Eg. Returns (0,0) when we're in top-left square and (2,2) when we're in bottom right
        return row // 3, col // 3

    def check_if_valid(self, number, row, col):
        # Check if number already exists in this column
        if number in [row[col] for row in self.board_to_solve]:
            return False

        # Check if number already exists in this row
        if number in self.board_to_solve[row][:]:
            return False

        # Check if number exists in this 3x3 square
        row_offset, col_offset = self.which_square(row, col)
        for i in range(3):
            for j in range(3):
                if self.board_to_solve[i + 3 * row_offset][j + 3 * col_offset] == number:
                    return False
        return True

    def solve(self):
        find = self.find_next_empty_cell()
        if not find:
            return True
        else:
            row, col = find
        for i in range(1, 10):
            if self.check_if_valid(i, row, col):
                self.board_to_solve[row][col] = i
                if self.solve():
                    return True
                self.board_to_solve[row][col] = 0
        return False


def draw_on_warped(recognized_board, solved_board, warped_image):
    im_height = int(warped_image.shape[0])
    im_width = int(warped_image.shape[1])
    vertical_step = im_height//9
    horizontal_step = im_width//9
    warped_text = warped_image

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.9
    color = (150, 220, 0)
    thickness = 1

    for row in range(9):
        for col in range(9):
            if recognized_board[row, col] == 0:
                org = ((col) * horizontal_step + ((horizontal_step//4)), (row+1) * vertical_step - ((vertical_step//4)-2))
                text = str(solved_board[row, col])
                if text == '0':
                    text = 'x'
                warped_text = cv2.putText(warped_image, text, org, font,
                                          fontScale, color, thickness, cv2.LINE_AA)
    return warped_text

def unwarp(warped, M, img):
    restored = cv2.warpPerspective(warped, M, (img.shape[1], img.shape[0]), flags=16)
    ret, mask = cv2.threshold(restored, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    border_only = cv2.bitwise_and(img, mask_inv)
    unwarped = border_only + restored
    return unwarped

nn_model = load_model('weights/MNIST_74k_UCI.h5')

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
frame_area = int(frame.shape[0] * frame.shape[1])
freeze = False
freeze_counter = 0
recognize_again_counter = 0
recognized_correctly = False
save_plot = False
show_recognized = False
show_solved = False
warped_text = np.array([0])
i = 0
print(f'frame area : {frame_area}')


frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# size_wide = (2*frame_width, frame_height)
# combined_movie = cv2.VideoWriter('combined.avi', cv2.VideoWriter_fourcc(*'MJPG'), 20, size_wide)

while True:
    ret, frame = cap.read()
    # frame = np.rot90(frame, k=3)      # Some non-native webcam software rotates the camera image, uncomment to compensate for that
    grabbed_board = Grabber(frame, nn_model)
    pre_processed = grabbed_board.pre_process()
    corners, contour = grabbed_board.contours_processing(pre_processed)
    if len(contour.shape) >= 2 :
        contour_area = cv2.contourArea(contour)
        if contour_area / frame_area < 0.15 and recognized_correctly:
            recognize_again_counter += 1
            if recognize_again_counter > 20:
                freeze=False
                recognize_again_counter = 0
    else:
        pass
    if freeze == False:
        if len(contour.shape) >= 2:
            contour_area = cv2.contourArea(contour)
            if contour_area > 45000:
                print(f'area: {contour_area}')
            if contour_area/frame_area >= 0.22: # threshold value: 67584
                freeze_counter += 1
                if freeze_counter > 15:
                    freeze = True
                    print('FREEZE')
                    grabbed_board.markROI(corners)
                    grabbed_board.warp_image(corners)
                    grid_extracted = grabbed_board.eliminate_grid_lines()
                    digit_images, is_a_digit, rather_1 = grabbed_board.find_digits(grid_extracted)
                    board_in_numpy = grabbed_board.recognize_digits(digit_images, is_a_digit, rather_1)
                    print(board_in_numpy)
                    recognized_board = board_in_numpy.copy()
                    board_solver = Solver(board_in_numpy)

                    if not board_solver.check_if_board_valid():
                        freeze = False
                        freeze_counter = 0
                    else:
                        i +=1
                        recognized_correctly = True
                        print(board_solver.check_if_board_valid())
                        board_solver.solve()
                        solved_board = board_solver.board_to_solve
                        warped_text = draw_on_warped(recognized_board, solved_board, grabbed_board.img_warped)
                        print(solved_board)


    grabbed_board.markROI(corners)
    grabbed_board.warp_image(corners)
    grid_extracted = grabbed_board.eliminate_grid_lines()

    cv2.imshow('Corners', grabbed_board.img_to_paint_on)
    cv2.imshow('Warped', grabbed_board.img_warped)
    combined_movie.write(grabbed_board.img_warped)

    cv2.imshow('Grid extracted', grid_extracted)

    try:
        warped_text = draw_on_warped(recognized_board, solved_board, grabbed_board.img_warped)

    except NameError:
        print('Board is yet to be recognized')

    if warped_text.size > 1:
        unwarped = unwarp(warped_text, grabbed_board.trans_matrix, grabbed_board.img)
        combined = cv2.hconcat([grabbed_board.img_to_paint_on, unwarped])
        cv2.imshow('Unwarped', unwarped)
        cv2.imshow('Combined', combined)
        cv2.imshow('Grid extracted', grid_extracted)

        # Save video
        #combined_movie.write(combined)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
