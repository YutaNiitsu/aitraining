import os
from image_crawl import ImageCrawl
from image_learn import ImageLearn
from image_evaluate import ImageEvaluate
from check_timestamp import check_timestamp_update

class ImageLearnApp:
    def crawl(self):
        self.img_crawl = ImageCrawl()
        self.img_crawl.crawl()

    def learn(self):
        self.img_learn = ImageLearn()
        self.img_learn.createModel()
        self.img_learn.learn()
        self.img_learn.save_model()

    def evaluate(self):
        self.img_val = ImageEvaluate()
        self.img_val.readModel()
        self.img_val.evaluate()
    

    
    def run(self):
        self.learn()
        self.evaluate()
        #if check_timestamp_update():
            #self.crawl()
            #self.learn()
            #self.evaluate()