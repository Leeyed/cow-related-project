from __future__ import print_function
import os

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET


def parse_xml(path):
    tree = ET.parse(path)
    # save the prefix
    line = [path.split(os.sep)[-1].split('.')[0]]
    for barcode in tree.iter('Barcode'):
        label = barcode.attrib['Type']
        # value = barcode.find('Value').text
        for point in barcode.iter('Point'):
            line.append(str(int(float(point.attrib['X']))))
            line.append(str(int(float(point.attrib['Y']))))
        line.append(label)
        # line.append(value)
    # print(line)
    line_text = ','.join(line)
    print(line_text)
    return line_text


url = 'D:\PycharmProjects\lius\yolov3\dataset\ZVZ-real-512\Markup'
jsons = os.listdir(url)
jsons_url = [os.path.join(url, json_url) for json_url in jsons]
# 读取json文件内容,返回字典格式
f = open("data.txt", 'w')
for json_url in jsons_url:
    line_text = parse_xml(json_url)
    f.write(line_text+'\n')
f.close()