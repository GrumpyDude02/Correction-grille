import pymupdf,cv2,numpy as np
from grid import Grid

class PDFFile:
    def __init__(self, path)->None:
        """constructeur"""
        try:
            self.file = pymupdf.open(path)
        except (FileExistsError,FileNotFoundError):
            raise FileNotFoundError
        self._raw_images_dict=[]
        self.grids:list[Grid]= []
        self.current_grid = None
        
    def extract_grids(self):
        self.extract_images()
        images=self.get_cv_images()
        if not self.grids:
            for image in images:
                self.grids.append(Grid(image))
    
    def get_current_grid(self):
        if not self.grids:
            return None
        if self.current_grid is None:
            self.current_grid = 0
        return self.grids[self.current_grid]
    
    def get_next_grid(self):
        if not self.grids:
            return None
        if self.current_grid is None:
            self.current_grid = 0
        else:
            self.current_grid +=1 
            self.current_grid %= len(self.grids)
        
        return self.grids[self.current_grid]
    
    def get_previous_grid(self):
        if not self.grids:
            return None
        if self.current_grid is None:
            self.current_grid = 0
        else:
            self.current_grid -=1 
            self.current_grid %= len(self.grids)
            
        return self.grids[self.current_grid]
        
    
    def extract_images(self):
        """returns a list of dictionnaries containing the raw image data in bytes 
        and other attributes read pymupdf.Document.extract_image for more information"""
        if self._raw_images_dict:
            return self._raw_images_dict
        biggest_image_dict = {'height':0}
        for i in range(len(self.file)):
            images = self.file.load_page(i).get_images(True)
            for image in images:
                base_image_dict = self.file.extract_image(xref=image[0])
                if base_image_dict["height"]>biggest_image_dict["height"]:
                    biggest_image_dict = base_image_dict
            self._raw_images_dict.append(biggest_image_dict)
        return self._raw_images_dict.copy()
            
    def save_images(self):
        if not self._raw_images_dict:
            self.extract_images()
        for i,image in enumerate(self._raw_images_dict):
            f=open(f"images/img-{i}.{image['ext']}","wb")
            f.write(image["image"])
            f.close()
    
    def get_cv_images(self):
        if not self._raw_images_dict:
            self.extract_images()
        images = []
        for image in self._raw_images_dict:
            image_array = np.frombuffer(image["image"], dtype=np.uint8) #8bit image
            images.append(cv2.imdecode(image_array, cv2.IMREAD_COLOR))
        return images
    
    def __del__(self):
        self.file.close()