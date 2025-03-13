import convert_pdf as PDF
import cv2
import numpy as np
import imutils
from statistics import mode
import tools

# TODO: increase contrast
# TODO: [[maybe]]: perspective correction
    
"""def checked(inv_binary_img,row):
    max_col_index = [0,1]
    for col_index,cell in enumerate(row):
        x,y,w,h=cell
        roi = inv_binary_img[y:y+h,x:x+w]
        black_pixels = cv2.countNonZero(roi)
        if max_col_index[0]<black_pixels:
            max_col_index=[black_pixels,col_index+1]
    return max_col_index


def save_rows_to_excel(rows,inv_binary_img , file_name="output.xlsx"):
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Extracted Data"
    #modes = get_row_modes_white_pixels(inv_binary_img,rows)
    for row_index, row in enumerate(rows, start=1):
        sheet.cell(row=row_index, column=checked(inv_binary_img,row)[1]).value = "cochée"
            #sheet.cell(row=row_index, column=col_index).value = f'{cell[0]},{cell[1]}'
    workbook.save(file_name)
"""

def process_pdf():

#------------------------------Pretraitement---------------------------------

    file = PDF.PDFFile("C:\\Users\\Youssef Bakouch\\Desktop\\grille_2.pdf")
    #file = PDF.PDFFile("D:\\projects\\Projet_tech\\documents\\testgrid2.pdf")
    images = file.get_cv_images()

    image_index = 0
    original_image=images[image_index]
    #original_image = imutils.rotate_bound(images[image_index],90)
    start = original_image.shape[1] // 2
    
    k_size = 7
    
    qcd = cv2.QRCodeDetector()
    
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_size))
    square_kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    square_kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    


    gray_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    binary_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 5)
    inverted_img = cv2.bitwise_not(binary_img.copy())
    
    decoded_info, points, data = qcd.detectAndDecode(original_image)
    
    horizontal_lines = cv2.morphologyEx(inverted_img,cv2.MORPH_ERODE,horizontal_kernel,iterations=8)
    horizontal_lines = cv2.morphologyEx(horizontal_lines,cv2.MORPH_DILATE,horizontal_kernel,iterations=14)

    vertical_lines = cv2.morphologyEx(inverted_img,cv2.MORPH_ERODE,vertical_kernel,iterations=8)    
    vertical_lines = cv2.morphologyEx(vertical_lines,cv2.MORPH_DILATE,vertical_kernel,iterations=8)

    combined_lines = cv2.bitwise_or(horizontal_lines, vertical_lines)
    
    cv2.imwrite("temp/combined_line.png",imutils.resize(combined_lines, width=combined_lines.shape[1] // 3))

#------------------------------tri des cellules selon les coordonnées---------------------------------
    rhs_combined_lines = combined_lines[:,start:]
    contours, _ = cv2.findContours(
        rhs_combined_lines , cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    
    rects =[] 
    for i in range(len(contours)):
        approximate_points = cv2.approxPolyDP(contours[i], 0.02* cv2.arcLength(contours[i], True), True)
        if len(approximate_points)==4:
            rects.append(contours[i])
    
    biggest_cnt=max(rects,key=lambda x: cv2.contourArea(x))
    
    x,y,w,h = cv2.boundingRect(biggest_cnt)
    
    cropped_combined_lines = rhs_combined_lines[y:y+h,x:x+w]
    
    cropped_combined_lines = cv2.ximgproc.thinning(cropped_combined_lines)
    
    inner_contours, _ = cv2.findContours(
        cropped_combined_lines , cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    
    areas = [cv2.contourArea(contour) for contour in inner_contours]
    median = np.median(areas)
    rows = tools.find_cells(inner_contours,0.45,areas,median)
    
#----------------------------sauvagrdes des etats des cellules dans un fichier excel--------------------------
    x1,y1,w1,h1 = rows[0][0]
    no_lines = cv2.absdiff(inverted_img,combined_lines)
    

    rhs_no_lines = no_lines[:,start:]
    cropped_no_lines = rhs_no_lines[y+y1:y+h,x+x1:x+w]
    
    cropped_no_lines = cv2.morphologyEx(cropped_no_lines,cv2.MORPH_OPEN,square_kernel3,iterations=1)
    dilated_cropped_no_lines = cv2.morphologyEx(cropped_no_lines,cv2.MORPH_DILATE,square_kernel3,iterations=6)
    
    checkmark_contours, _ =cv2.findContours(
        dilated_cropped_no_lines , cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    
    checkmark_bboxes=[]
    
    for contour in checkmark_contours:
        if cv2.contourArea(contour)>900:
            checkmark_bboxes.append(cv2.boundingRect(contour))
    
    checked_cells = [[0 for _ in range(len(rows[0]))] for _ in rows]
    
    temp = []
    for row in rows:
        temp.append(tools.add_offset_bbox(row,(-x1,-y1)))
    
    for bbox in checkmark_bboxes:
        xc,yc,w,h=bbox
        tools.assign_checkmarks_with_voting(bbox,temp,checked_cells)
        cv2.rectangle(original_image,(xc+x+x1+start,yc+y+y1),(xc+x+x1+start+w,yc+y+y1+h),(0,0,255),2)

    
    
    #tools.save_to_excel(tools.add_offset_bbox(rows,(-x1,-y1)),cropped_no_lines)
    
#----------------------debug purposes-----------------------------
    offset_rows = []
    for row in rows:
        offset_rows.append(tools.add_offset_bbox(row,(x+start,y)))
    
    #tools.save_cell_imgs(cropped_no_lines,rows,"inverted_cells")
    
    tools.draw_bounding_boxes(original_image,offset_rows)
    cv2.imwrite("temp/og.png",imutils.resize(original_image, width=original_image.shape[1] // 3))
    cv2.imwrite("temp/inverted.png",imutils.resize(inverted_img, width=original_image.shape[1] // 3))
    cv2.imwrite("temp/binary.png",imutils.resize(binary_img, width=original_image.shape[1] // 3))
    tools.draw_bounding_boxes(original_image,offset_rows)
    cv2.imwrite("temp/roi.png",imutils.resize(original_image, width=original_image.shape[1] // 3))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return (original_image,tools.decode_qr(decoded_info),checked_cells)
    

if __name__=="__main__":
    process_pdf()