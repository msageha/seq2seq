import datetime
import numpy as np
from chainer import Chain, Variable, cuda, optimizer, optimizers, serializers
import chainer.functions as F
import chainer.links as L
import argparse
import json
import os

from model import AttSeq2Seq
from model import DataConverter

def load_data():
    print('start load_data')
    data = []
    for file in os.listdir('../data/under20'):
        with open('../data/under20/{}'.format(file)) as f:
            for line in f:
                input_text = line[7:].strip()
                line = f.readline()
                output_text = line[8:].strip()
                data.append([input_text, output_text])
    return data

def training():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_size', type=int, default=200)
    parser.add_argument('--dropout', '-d', type=float, default=0.2)
    parser.add_argument('--batch_size', '-b', type=int, default=15)
    parser.add_argument('--batch_col_size', type=int, default=20)
    parser.add_argument('--epoch', '-e', type=int, default=50)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--out', '-o', default='result', help='Directory to output the result')
    parser.add_argument('--model', '-m', default='', type=str)
    args = parser.parse_args()

    print(json.dumps(args.__dict__, indent=2))

    # GPUのセット
    if args.gpu > -1: # numpyかcuda.cupyか
        xp = cuda.cupy
        cuda.get_device(args.gpu).use()
    else:
        xp = np
    # 教師データ
    data = load_data()
    N = len(data) # 教師データの数

    # 教師データの読み込み
    print('initialize DataConverter')
    data_converter = DataConverter(batch_col_size=args.batch_col_size) # データコンバーター
    data_converter.load(data) # 教師データ読み込み

    model = AttSeq2Seq(input_size=200, hidden_size=args.hidden_size, batch_col_size=args.batch_col_size, dropout=args.dropout, device=args.gpu)

    if args.gpu >= 0:
        model.to_gpu(0)
    if args.model != '':
        serializers.load_npz(args.model, model)

    opt = optimizers.Adam()
    opt.setup(model)
    opt.add_hook(optimizer.GradientClipping(5))

    model.reset()

    # 学習開始
    print("Train start")
    st = datetime.datetime.now()
    model_file_name = str(st)[:-7]
    for epoch in range(args.epoch):
        # ミニバッチ学習
        perm = np.random.permutation(N) # ランダムな整数列リストを取得
        total_loss = 0
        for i in range(0, N, args.batch_size):
            enc_words = data_converter.train_queries[perm[i:i+args.batch_size]]
            dec_words = data_converter.train_responses[perm[i:i+args.batch_size]]
            model.reset()
            loss = model(enc_words=enc_words, dec_words=dec_words, train=True)
            loss.backward()
            loss.unchain_backward()
            total_loss += loss.data
            opt.update()
            print('{0}/{1}:'.format(i, N), end='\t', flush=True)
        #output_path = "./att_seq2seq_network/{}_{}.network".format(epoch+1, total_loss)
        #serializers.save_npz(output_path, model)
        ed = datetime.datetime.now()
        print("\nepoch:\t{0}\ttotal loss:\t{1}\ttime:\t{2}".format(epoch+1, total_loss, ed-st))
        st = datetime.datetime.now()
        model.to_cpu()
        serializers.save_npz("model/{0}_epoch-{1}.npz".format(model_file_name, epoch+1), model) # npz形式で書き出し
        model.to_gpu()

if __name__ == '__main__':
    training()
