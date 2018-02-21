import datetime
import numpy as np
from chainer import Chain, Variable, cuda, optimizer, optimizers, serializers
import chainer.functions as F
import chainer.links as L
import MeCab
import argparse 

from model import AttSeq2Seq
from model import DataConverter


def predict(model, query):
    enc_query = data_converter.sentence2vectors(query, train=False)
    dec_response = model(enc_words=enc_query, train=False)
    response = data_converter.vectors2sentences(dec_response)
    print(query, "=>", response)
 

if __name__ == "__main__":
    batch_col_size = 12
    data_converter = DataConverter(batch_col_size=batch_col_size) # データコンバーター
    print('loaded')
    model = AttSeq2Seq(input_size=200, hidden_size=200, batch_col_size=batch_col_size)
    serializers.load_npz("mymodel.npz", model)

    predict(model, "初めまして。")
    predict(model, "どこから来たんですか？")
    predict(model, "日本のどこに住んでるんですか？")
    predict(model, "仕事は何してますか？")
    predict(model, "お会いできて嬉しかったです。")
    predict(model, "おはよう。")
    predict(model, "いつも何時に起きますか？")
    predict(model, "朝食は何を食べますか？")
    predict(model, "朝食は毎日食べますか？")
    predict(model, "野菜をたくさん取っていますか？")
    predict(model, "週末は何をしていますか？")
    predict(model, "どこに行くのが好き？")