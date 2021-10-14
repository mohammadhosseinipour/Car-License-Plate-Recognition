import os
import time
import urllib
import requests
import magic
import progressbar
from urllib.parse import quote
import os
import urllib.request as ulib
from bs4 import BeautifulSoup as Soup
import json

class simple_image_download:
    def __init__(self):
        pass

    def urls(self, keywords, limit, extensions={'.jpg', '.png', '.ico', '.gif', '.jpeg'}):
        keyword_to_search = [str(item).strip() for item in keywords.split(',')]
        i = 0
        links = []

        things = len(keyword_to_search) * limit

        bar = progressbar.ProgressBar(maxval=things, \
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()]).start()

        while i < len(keyword_to_search):
            url = 'https://www.google.com/search?q=' + quote(
                keyword_to_search[i].encode(
                    'utf-8')) + '&biw=1536&bih=674&tbm=isch&sxsrf=ACYBGNSXXpS6YmAKUiLKKBs6xWb4uUY5gA:1581168823770&source=lnms&sa=X&ved=0ahUKEwioj8jwiMLnAhW9AhAIHbXTBMMQ_AUI3QUoAQ'
            raw_html = self._download_page(url)

            end_object = -1;
            google_image_seen = False;
            j = 0

            while j < limit:
                while (True):
                    try:
                        new_line = raw_html.find('"https://', end_object + 1)
                        end_object = raw_html.find('"', new_line + 1)

                        buffor = raw_html.find('\\', new_line + 1, end_object)
                        if buffor != -1:
                            object_raw = (raw_html[new_line + 1:buffor])
                        else:
                            object_raw = (raw_html[new_line + 1:end_object])

                        if any(extension in object_raw for extension in extensions):
                            break

                    except Exception as e:
                        break


                try:
                    r = requests.get(object_raw, allow_redirects=True, timeout=1)
                    if('html' not in str(r.content)):
                        mime = magic.Magic(mime=True)
                        file_type = mime.from_buffer(r.content)
                        file_extension = f'.{file_type.split("/")[1]}'
                        if file_extension == '.png' and not google_image_seen:
                            google_image_seen = True
                            raise ValueError();
                        links.append(object_raw)
                        bar.update(bar.currval + 1)
                    else:
                        j -= 1
                except Exception as e:
                    j -= 1
                j += 1

            i += 1

        bar.finish()
        return(links)

    def save_images(self,links, search_name):
        directory = search_name.replace(' ', '_')
        if not os.path.isdir(directory):
            os.mkdir(directory)

        for i, link in enumerate(links):
            savepath = os.path.join(directory, '{:06}.png'.format(i))
            try:
                ulib.urlretrieve(link, savepath)
            except:
                pass




    def _create_directories(self, main_directory, name):
        name = name.replace(" ", "_")
        try:
            if not os.path.exists(main_directory):
                os.makedirs(main_directory)
                time.sleep(0.2)
                path = (name)
                sub_directory = os.path.join(main_directory, path)
                if not os.path.exists(sub_directory):
                    os.makedirs(sub_directory)
            else:
                path = (name)
                sub_directory = os.path.join(main_directory, path)
                if not os.path.exists(sub_directory):
                    os.makedirs(sub_directory)

        except OSError as e:
            if e.errno != 17:
                raise
            pass
        return

    def _download_page(self,url):

        try:
            headers = {}
            headers['User-Agent'] = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.90 Safari/537.36"
            req = urllib.request.Request(url, headers=headers)
            resp = urllib.request.urlopen(req)
            respData = str(resp.read())
            return respData

        except Exception as e:
            print(e)
            exit(0)


a=simple_image_download()
# print(a.urls("car",4))
# a.save_images(a.urls("ماشین با پلاک ایرانی",500),"car1")
a.save_images(a.urls("car in street with license plate",500),"car3")
