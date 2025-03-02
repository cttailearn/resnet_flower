import requests
import argparse

# url和端口携程自己的
flask_url = 'http://127.0.0.1:5012/predict'


def predict_result(image_path):
    #啥方法都行
    image = open(image_path, 'rb').read()
    payload = {'image': image}
    #request发给server.
    r = requests.post(flask_url, files=payload).json()
    
    # 成功的话在返回.
    if r['success']:
        # 输出结果.
        for (i, result) in enumerate(r['predictions']):
            print('{}. {}. {}: {:.4f}'.format(i + 1, result['label'], result['label_name'],
                                          result['probability']))
    # 失败了就打印.
    else:
        print('Request failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classification demo')
    parser.add_argument('--file', default='./test/image_06753.jpg', type=str, help='输入一张花朵的图片')

    args = parser.parse_args()
    predict_result(args.file)
