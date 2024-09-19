import chardet

# Detect the encoding of the file
with open('/FloodsInPakistan-tweets.csv', 'rb') as rawdata:
    result = chardet.detect(rawdata.read(10000))

print(result)
