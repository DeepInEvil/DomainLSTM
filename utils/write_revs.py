#script for creating the positive and negative reviews for each domain, use the json_file name, and the domain name as input
import sys
import json

file_n = sys.argv[1]
domain = sys.argv[2]
out_p = '../data/'+domain
positive = open(out_p + '/positive.csv', 'w')
negative = open(out_p + '/negative.csv','w')
with open(file_n, 'r') as f:
	for lines in f:
		data = json.loads(lines)
		score = data['overall']
		if score > 3:
			positive.write(data['reviewText'])
			positive.write('\n')
		elif score < 3:
			negative.write(data['reviewText'])					
			negative.write('\n')

	
