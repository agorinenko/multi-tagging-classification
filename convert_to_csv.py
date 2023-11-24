import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
"""
Преобразует xml документы в csv

Stackoverflow data set:
    https://archive.org/details/stackexchange
"""

if __name__ == '__main__':
    # file = 'data/ru.stackoverflow.com/Posts.xml'
    # output = 'data/stackoverflow_posts.csv'
    file = 'data/russian.stackexchange.com/Posts.xml'
    output = 'data/stackexchange_posts.csv'
    data = []
    columns = ('Id', 'Title', 'Body', 'Tags')
    for event, elem in ET.iterparse(file, events=('start', 'end')):
        if elem.tag == 'row':
            row = []
            for name in columns:
                row.append(elem.attrib.get(name))
            data.append(row)
            
    df = pd.DataFrame(data, columns=columns)
    df.set_index([columns[0]], inplace=True)
    df.to_csv(output)