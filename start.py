import requests
import json 
import ssl
import csv

# # ssl._create_default_https_context = ssl._create_unverified_context
# url = "http://localhost:9655/hitec/classify/domain/tweets/lang/en"

PAYLOAD_TWEETS_EN = """
    [
        {"text": "@MyTeleC today is the first day for my billing period. I had mobile data turned off since last night. How did I loose 500kbyes? https://t.co/VUZV4moYhr"},
        {"text": "@MyTeleC I tried Vodafones broadband... Constant drop outs made it unusable, speed was ok when it worked.I went crawling back to sky within 2 months...The best thing about Vodafone broadband is its easy to cancel!"},
        {"text": "@MyTeleC why isit impossible to speak to one of your advisors and when I do manage to get through to someone im cut of. Im trying to pay my bill for god sake. Do you want it or not?!"},
        {"text": "@MyTeleC hey just wondering do you offer 4G calling in these two postcodes ng175be and this one Ng174ad? Can't find out on the site"},
        {"text": "@MyTeleC I understand this and was told last night, I can terminate from the end of June. You really don't understand customer loyalty/retention. Again, shocking"},
        {"text": "@MyTeleC @JohnBoyega @touchithighnote the vid"}
    ]
"""
# file = open('./data/data.csv', 'r', encoding='utf-8')
# # data = json.dumps(PAYLOAD_TWEETS_EN)
# # data = json.dump(file)
# print(file)

# # r = requests.post(url, PAYLOAD_TWEETS_EN, verify=False)
# # print(r.url)
# # print(r.text)
def getTweetInJson(path_original, path_json):
    csvFile = open(path_original, 'r', encoding= 'utf-8')
    jsonFile = open(path_json, 'w', encoding = 'utf-8')

    reader = csv.reader(csvFile)
    next(reader)
    temp_list = []
    for row in reader:
        # temp_list.append(row[4])
        # json.dump("text",jsonFile)
        # json.dump(":",jsonFile)
        # json.dump(row[4],jsonFile)
        # json.dump({"text":row[4]},jsonFile)
        # jsonFile.write(',\n')
        temp_list.append({"text":row[4]})
    json_list = json.dumps(temp_list)
    jsonFile.write(json_list)
    return json_list

    # return temp_list
def changeJsonToStr(path_json):
    # jsonFile = open(path_json, 'r', encoding = 'utf-8')
    # temp_list = list(jsonFile)
    # temp_str = str(temp_list)
    # # print(temp_str)
    jsonFile = open(path_json, 'r', encoding = 'utf-8')
    data = str(json.load(jsonFile))
    # data = '"""'+str(data)+'"""'
    # print(data)
    # return temp_str
    # return data
    print(data)

def visitClassifier(url, json_file):
    # data = json.dumps(json_file)
    jsonFile = open(json_file, 'r', encoding = 'utf-8')
    data = json.load(jsonFile)
    # print(data)
    r = requests.post(url, data, verify=False)
    return r.text

PAYLOAD_TEST_EN_SINGLE = """[{"text": "@samirdayalsingh @messenger if you dont use facebook, and you are still a founder then please submit your details!!!!! "}]"""

PAYLOAD_TEST_EN = """[{"text": "@samirdayalsingh @messenger if you dont use facebook, and you are still a founder then please submit your details!!!!! "}, {"text": "@piratefund @messenger but i appreciate the intent."}, {"text": "@piratefund @messenger when youre saying founders first, you dont get to share a facebook link. we dont use facebook."}, {"text": "@pagesmanagerw8 @facebook @messenger can we expect pages to appear in a dictionary order while switching pages in p"}]"""
PAYLOAD_STR = """[{"text": "@Op1 buongiorno ho scritto ieri un dm nessuna risposta""}, {"text": "@Op1 buongiorno ho scritto ieri un dm nessuna risposta"}, {"text": "@Op1 buongiorno ho scritto ieri un dm nessuna risposta"}]"""
if __name__ == "__main__":
    
    url = "http://localhost:9655/hitec/classify/domain/tweets/lang/en"
    path_ORG = './data/data.csv'
    path_ORG_Testset = './data/raw_test_new.csv'
    path_ORG_Testset_300 = './data/raw_test_new_300.csv'
    path_JSON = './data/data.json'
    path_JSON_2 = './data/data_2.json'
    path_JSON_3 = './data/test_json.json'
    path_JSON_4 = './data/data_4.json'
    path_JSON_Testset = './data/testset.json'
    path_JSON_result_testset = './data/result_testset.json'
    path_ORG_2 = './data/data_remove_RT.csv'
    
    # print(getTweetInJson(path_ORG, path_JSON_2))
    # print(changeJsonToStr('./data/data_2.json'))
    # print(visitClassifier(url, path_JSON))
    # print(changeJsonToStr(path_JSON_3))
    # print(changeJsonToStr(path_JSON_3))
    # post_str = """[{'text': '@Op1 buongiorno ho scritto ieri un dm nessuna risposta'}, {'text': '@Op1 buongiorno ho scritto ieri un dm nessuna risposta!'}, {'text': '@Op1 buongiorno ho scritto ieri un dm nessuna risposta'}, {'text': '@Op1 buongiorno ho scritto ieri un dm nessuna risposta'}]"""

    # print(requests.post(url, PAYLOAD_TEST_EN, verify=False).text)
    # changeJsonToStr(path_JSON_3)
    # print(changeJsonToStr(path_JSON_3))

     # print(requests.post(url, getTweetInJson(path_ORG, path_JSON_2), verify=False).text)


    # print(requests.post(url, getTweetInJson(path_ORG_2, path_JSON_4), verify=False).text)

    # print(requests.post(url, PAYLOAD_TEST_EN_SINGLE, verify=False).text)

    # getTweetInJson(path_ORG_Testset, path_JSON_Testset) #get testset.json
    print(requests.post(url, getTweetInJson(path_ORG_Testset_300, path_JSON_Testset), verify=False).text) #get testset result















