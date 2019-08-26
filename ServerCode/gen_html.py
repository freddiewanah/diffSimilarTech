from yattag import Doc
import glob
import json

# get all the pic files in the current folder
picDir = './res/*png'
picList = glob.glob(picDir)

def get_json(pic):
	jsonDir = pic.replace('png','json')
	return jsonDir
	
# prepare the html
doc, tag, text = Doc().tagtext()

doc.asis('<!DOCTYPE html>')
with tag('html'):
	for pic in picList:
		with tag('div', id='photo-container'):
			js = get_json(pic)
			with open(js) as jFile:
				pic_des = json.load(jFile)
    			doc.stag('img', src=pic, klass="photo")
			
print(doc.getvalue())