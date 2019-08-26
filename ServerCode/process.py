import xmltodict
import glob
import json



# read res xml files
xmlDir = './res/layout/*xml'
xmlList = glob.glob(xmlDir)
libs = {}
for xml in xmlList:
	with open(xml) as fd:
		doc = xmltodict.parse(fd.read())
		# find the 3rd party libraries within xml files
		for k in doc.keys():
			visited = [{k:doc[k]}]
			while len(visited) > 0:
				key, nodes = visited.pop()

				# check if node is valid to add to libs
				
				for node_key in node.keys():
					children = node[node_key]
# 					if isinstance(node, list):
				
# read json files through id and match 
jsonDir = './states/*json'
jsonList = glob.glob(jsonDir)
for js in jsonList:
	with open(js) as fd:
		state = json.load(fd)
		for view in state['views']:
			if() 
# save the img and json view
# https://stackoverflow.com/questions/37435369/matplotlib-how-to-draw-a-rectangle-on-image
# select the certain area from img and draw with np

