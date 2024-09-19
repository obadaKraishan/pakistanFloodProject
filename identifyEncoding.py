import chardet

# Detect the encoding of the file
with open('/Volumes/Kraishan 1/TTU//Thesis/NEW/Pakistan/FloodsInPakistan-tweets.csv', 'rb') as rawdata:
    result = chardet.detect(rawdata.read(10000))

print(result)
