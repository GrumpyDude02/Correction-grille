import cv2, numpy as np
import imutils
import tools


class Grid:

    k_size = 7
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_size))
    square_kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    tolerance = 0.45
    save = True

    def __init__(self, cv_image: np.ndarray):

        # to change, absolutely
        if cv_image.shape[1] > cv_image.shape[0]:
            self.original_matrix = imutils.rotate_bound(cv_image, 90)
        else:
            self.original_matrix = cv_image

        self.drawn_og_img = self.original_matrix.copy()
        self.middle_x = self.original_matrix.shape[1] // 2
        self.qr_code_data = None
        self.gray_img = None
        self.binary_img = None
        self.inverted_img = None
        self.combined_lines = None
        self.no_lines = None
        self.sorted_cells = None
        self.checked_cells = None

    def _decode_qr_code(self):
        qcd = cv2.QRCodeDetector()
        decoded_info, points, data = qcd.detectAndDecode(self.original_matrix)
        self.qr_code_data = tools.decode_qr(decoded_info)

    def _preprocess(self):
        self._decode_qr_code()
        self.gray_img = cv2.cvtColor(self.original_matrix, cv2.COLOR_BGR2GRAY)
        self.binary_img = cv2.adaptiveThreshold(
            self.gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 5
        )
        self.inverted_img = cv2.bitwise_not(self.binary_img.copy())

    def _isolate_lines(self):
        self.horizontal_lines = cv2.morphologyEx(
            self.inverted_img, cv2.MORPH_ERODE, Grid.horizontal_kernel, iterations=8
        )
        self.horizontal_lines = cv2.morphologyEx(
            self.horizontal_lines, cv2.MORPH_DILATE, Grid.horizontal_kernel, iterations=14
        )

        self.vertical_lines = cv2.morphologyEx(
            self.inverted_img, cv2.MORPH_ERODE, Grid.vertical_kernel, iterations=8
        )
        self.vertical_lines = cv2.morphologyEx(
            self.vertical_lines, cv2.MORPH_DILATE, Grid.vertical_kernel, iterations=8
        )

        self.combined_lines = cv2.bitwise_or(self.horizontal_lines, self.vertical_lines)

    def _extract_cells(self):
        rhs_combined_lines = self.combined_lines[:, self.middle_x :]
        outer_contours, _ = cv2.findContours(
            rhs_combined_lines, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        rects = []
        for i in range(len(outer_contours)):
            approximate_points = cv2.approxPolyDP(
                outer_contours[i], 0.02 * cv2.arcLength(outer_contours[i], True), True
            )
            if len(approximate_points) == 4:
                rects.append(outer_contours[i])

        self.bbox_biggest_rect = cv2.boundingRect(
            max(rects, key=lambda x: cv2.contourArea(x))
        )

        x, y, w, h = self.bbox_biggest_rect
        self.cropped_combined_lines = rhs_combined_lines[y : y + h, x : x + w]
        cv2.imwrite("temp/gsf.png", self.cropped_combined_lines)
        self.cropped_combined_lines = cv2.ximgproc.thinning(self.cropped_combined_lines)

        inner_contours, _ = cv2.findContours(
            self.cropped_combined_lines, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        bboxes = []

        areas = [cv2.contourArea(contour) for contour in inner_contours]
        median = np.median(areas)

        for i in range(len(inner_contours)):
            approximate_points = cv2.approxPolyDP(
                inner_contours[i], 0.1 * cv2.arcLength(inner_contours[i], True), True
            )
            if len(approximate_points) == 4 and median * (1 - Grid.tolerance) < areas[
                i
            ] < median * (1 + Grid.tolerance):
                bboxes.append(cv2.boundingRect(inner_contours[i]))

        self.sorted_cells = tools.sort_cells(bboxes)

    def _isolate_checkmarks(self):
        x, y, w, h = self.sorted_cells[0][0]
        x1, y1, w1, h1 = self.bbox_biggest_rect

        self.no_lines = cv2.absdiff(self.inverted_img, self.vertical_lines)

        rhs_no_lines = self.no_lines[:, self.middle_x:]
        self.cropped_no_lines = rhs_no_lines[y1 : y1 + h1, x1 : x1 + w1][y:, x:]

        self.cropped_no_lines = cv2.morphologyEx(
            self.cropped_no_lines, cv2.MORPH_OPEN, Grid.square_kernel3, iterations=1
        )
        
        dilated_cropped_no_lines = cv2.morphologyEx(
            self.cropped_no_lines, cv2.MORPH_DILATE, Grid.horizontal_kernel, iterations=8
        )
        
        
        dilated_cropped_no_lines = cv2.absdiff(dilated_cropped_no_lines,self.horizontal_lines[:, self.middle_x:][y1 : y1 + h1, x1 : x1 + w1][y:, x:])
        
        dilated_cropped_no_lines = cv2.bitwise_and(dilated_cropped_no_lines,self.inverted_img[:, self.middle_x:][y1 : y1 + h1, x1 : x1 + w1][y:, x:])
        
        dilated_cropped_no_lines =cv2.morphologyEx(dilated_cropped_no_lines,cv2.MORPH_CLOSE,Grid.square_kernel3,iterations=1)
        
        cv2.imwrite("temp/dileted_checked_marks.png", dilated_cropped_no_lines)
        
        self.cropped_no_lines = dilated_cropped_no_lines

        checkmark_contours, _ = cv2.findContours(
            dilated_cropped_no_lines, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        self.checkmark_bboxes = []

        self.checked_cells = [
            [0 for j in range(len(self.sorted_cells[i]))]
            for i in range(len(self.sorted_cells))
        ]

        temp = []
        for row in self.sorted_cells:
            temp.append(tools.add_offset_bbox(row, (-x, -y)))

        for contour in checkmark_contours:
            bbox = cv2.boundingRect(contour)
            if bbox[2]*bbox[3] > 500:
                self.checkmark_bboxes.append(bbox)
                tools.assign_checkmarks_with_voting(bbox, temp, self.checked_cells)
        
        for row in self.checked_cells:
            formatted_row = [f"{value:.2f}" for value in row]
            print("\t".join(formatted_row))

    def _draw_cells_bboxs(self):
        xcell, ycell, _, _ = self.sorted_cells[0][0]
        x, y, w, h = self.bbox_biggest_rect
        for bbox in self.checkmark_bboxes:
            xb, yb, wb, hb = bbox
            cv2.rectangle(self.cropped_no_lines,(xb,yb),(xb+wb,yb+hb),255,2)
            cv2.rectangle(
                self.drawn_og_img,
                (xb + x + self.middle_x + xcell, ycell + y + yb),
                (xb + x + self.middle_x + xcell + wb, ycell + y + yb + hb),
                (0, 0, 255),
                2,
            )

        for row in self.sorted_cells:
            for cell in row:
                xcell, ycell, wcell, hcell = cell
                cv2.rectangle(
                    self.drawn_og_img,
                    (xcell + self.middle_x, y + ycell),
                    (xcell + self.middle_x + wcell, y + ycell + hcell),
                    (0, 255, 0),
                    2,
                )

    def save_imgs(self, folder):
        if not Grid.save:
            return
        try:
            cv2.imwrite(f"{folder}/roi.png", self.drawn_og_img)
            cv2.imwrite(f"{folder}/og.png", self.original_matrix)
            cv2.imwrite(f"{folder}/inverted.png", self.inverted_img)
            cv2.imwrite(f"{folder}/binary.png", self.binary_img)
            cv2.imwrite(f"{folder}/checkmarks.png", self.cropped_no_lines)
        except (ValueError, FileExistsError, FileNotFoundError):
            print("Sauvegarde Impossible")

    def run_analysis(self):
        if not self.checked_cells:
            self._preprocess()
            self._isolate_lines()
            self._extract_cells()
            self._isolate_checkmarks()
            self._draw_cells_bboxs()
            self.save_imgs("temp")
        return {
            "image": self.drawn_og_img,
            "qr_code_info": self.qr_code_data,
            "checkmarks": self.checked_cells,
        }
