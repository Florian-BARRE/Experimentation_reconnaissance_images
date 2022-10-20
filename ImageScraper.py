from icrawler.builtin import BingImageCrawler
from concurrent.futures import ThreadPoolExecutor

def download_img(object: str, save_path: str, quantity: int):
    bing_crawler = BingImageCrawler(storage={'root_dir': save_path})
    bing_crawler.crawl(keyword=object, filters=None, max_num=quantity, offset=0)

qte = 400
with ThreadPoolExecutor(4) as executor:
    executor.submit(download_img, "human face", "./images/humans_faces", qte)

