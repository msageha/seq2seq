import numpy as np
from chainer import Chain, Variable, cuda, optimizer, optimizers, serializers
import chainer.functions as F
import chainer.links as L
import MeCab
import gensim

w2v_path = '..'

class DataConverter:
    def __init__(self, batch_col_size):
        # クラスの初期化
        # :param batch_col_size: 学習時のミニバッチ単語数サイズ
        self.mecab = MeCab.Tagger() # 形態素解析器
        self.batch_col_size = batch_col_size
        self.model = gensim.models.KeyedVectors.load_word2vec_format('{0}/entity_vector.model.txt'.format(w2v_path))

    def load(self, data):
        # 学習時に、教師データを読み込んでミニバッチサイズに対応したNumpy配列に変換する
        # :param data: 対話データ
        # 教師データのID化と整理
        queries, responses = [], []
        for d in data:
            query, response = d[0], d[1] #  エンコード文、デコード文
            queries.append(self.sentence2vectors(sentence=query, train=True, sentence_type="query"))
            responses.append(self.sentence2vectors(sentence=response, train=True, sentence_type="response"))
        self.train_queries = np.array(queries)
        self.train_responses = np.array(responses)

    def sentence2words(self, sentence):
        # 文章を単語の配列にして返却する
        # :param sentence: 文章文字列
        sentence_words = []
        for m in self.mecab.parse(sentence).split("\n"): # 形態素解析で単語に分解する
            w = m.split("\t")[0].lower() # 単語
            if len(w) == 0 or w == "eos": # 不正文字、EOSは省略
                continue
            sentence_words.append(w)
        sentence_words.append("EOS") # 最後にvocabに登録している&lt;eos&gt;を代入する
        return sentence_words

    def word2vector(self, word):
        if word and word in self.model.vocab:
            return self.model[word]
        else:
            return np.zeros(200, dtype=np.float32)

    def sentence2vectors(self, sentence, train=True, sentence_type="query"):
        # 文章を単語IDのNumpy配列に変換して返却する
        # :param sentence: 文章文字列
        # :param train: 学習用かどうか
        # :sentence_type: 学習用でミニバッチ対応のためのサイズ補填方向をクエリー・レスポンスで変更するため"query"or"response"を指定　
        # :return: 単語IDのNumpy配列
        vectors = [] # 単語IDに変換して格納する配列
        sentence_words = self.sentence2words(sentence) # 文章を単語に分解する
        for word in sentence_words:
            vector = self.word2vector(word)
            vectors.append(vector)
        # 学習時は、ミニバッチ対応のため、単語数サイズを調整してNumpy変換する
        if train:
            if sentence_type == "query": # クエリーの場合は前方にミニバッチ単語数サイズになるまで-1を補填する
                while len(vectors) > self.batch_col_size: # ミニバッチ単語サイズよりも大きければ、ミニバッチ単語サイズになるまで先頭から削る
                    vectors.pop(0)
                vectors = np.array([np.zeros(200)]*(self.batch_col_size-len(vectors))+vectors, dtype="float32")
            elif sentence_type == "response": # レスポンスの場合は後方にミニバッチ単語数サイズになるまで-1を補填する
                while len(vectors) > self.batch_col_size: # ミニバッチ単語サイズよりも大きければ、ミニバッチ単語サイズになるまで末尾から削る
                    vectors.pop()
                vectors = np.array(vectors+[np.zeros(200)]*(self.batch_col_size-len(vectors)), dtype="float32")
        else: # 予測時は、そのままNumpy変換する
            vectors = np.array([vectors], dtype="float32")
        return vectors

    def vectors2sentences(self, vectors):
        # 予測時に、単語IDのNumpy配列を単語に変換して返却する
        # :param ids: 単語IDのNumpy配列
        # :return: 単語の配列
        words = [] # 単語を格納する配列
        for vector in vectors: # 順番に単語IDを単語辞書から参照して単語に変換する
            word = self.model.similar_by_vector(cuda.to_cpu(vector), topn=1)[0][0]
            if word == 'EOS':
                break
            words.append(word)
            if word == '。':
                break
        return ''.join(words)

# モデルクラスの定義

# LSTMエンコーダークラス
class LSTMEncoder(Chain):
    def __init__(self, input_size, hidden_size):
        # Encoderのインスタンス化
        # :param vocab_size: 使われる単語の種類数
        # :param embed_size: 単語をベクトル表現した際のサイズ
        # :param hidden_size: 隠れ層のサイズ
        super(LSTMEncoder, self).__init__(
            eh = L.Linear(input_size, 4 * hidden_size),
            hh = L.Linear(hidden_size, 4 * hidden_size)
        )

    def __call__(self, x, c, h):
        # Encoderの計算
        # :param x: one-hotな単語
        # :param c: 内部メモリ
        # :param h: 隠れ層
        # :return: 次の内部メモリ、次の隠れ層
        e = F.tanh(x)
        return F.lstm(c, self.eh(e) + self.hh(h))

# Attention Model + LSTMデコーダークラス
class AttLSTMDecoder(Chain):
    def __init__(self, input_size, hidden_size):
        # Attention ModelのためのDecoderのインスタンス化
        # :param vocab_size: 語彙数
        # :param embed_size: 単語ベクトルのサイズ
        # :param hidden_size: 隠れ層のサイズ
        super(AttLSTMDecoder, self).__init__(
            eh = L.Linear(input_size, 4 * hidden_size), # 単語ベクトルを隠れ層の4倍のサイズのベクトルに変換する層
            hh = L.Linear(hidden_size, 4 * hidden_size), # Decoderの中間ベクトルを隠れ層の4倍のサイズのベクトルに変換する層
            fh = L.Linear(hidden_size, 4 * hidden_size), # 順向きEncoderの中間ベクトルの加重平均を隠れ層の4倍のサイズのベクトルに変換する層
            bh = L.Linear(hidden_size, 4 * hidden_size), # 順向きEncoderの中間ベクトルの加重平均を隠れ層の4倍のサイズのベクトルに変換する層
            he = L.Linear(hidden_size, input_size), # 隠れ層サイズのベクトルを単語ベクトルのサイズに変換する層
            ey = L.Linear(input_size, input_size) # 単語ベクトルを語彙数サイズのベクトルに変換する層
        )

    def __call__(self, y, c, h, f, b):
        # Decoderの計算
        # :param y: Decoderに入力する単語
        # :param c: 内部メモリ
        # :param h: Decoderの中間ベクトル
        # :param f: Attention Modelで計算された順向きEncoderの加重平均
        # :param b: Attention Modelで計算された逆向きEncoderの加重平均
        # :return: 語彙数サイズのベクトル、更新された内部メモリ、更新された中間ベクトル
        e = F.tanh(y) # 単語を単語ベクトルに変換
        c, h = F.lstm(c, self.eh(e) + self.hh(h) + self.fh(f) + self.bh(b)) # 単語ベクトル、Decoderの中間ベクトル、順向きEncoderのAttention、逆向きEncoderのAttentionを使ってLSTM
        t = self.ey(F.tanh(self.he(h))) # LSTMから出力された中間ベクトルを語彙数サイズのベクトルに変換する
        return t, c, h

# Attentionモデルクラス
class Attention(Chain):
    def __init__(self, hidden_size, device):
        # Attentionのインスタンス化
        # :param hidden_size: 隠れ層のサイズ
        super(Attention, self).__init__(
            fh = L.Linear(hidden_size, hidden_size), # 順向きのEncoderの中間ベクトルを隠れ層サイズのベクトルに変換する線形結合層
            bh = L.Linear(hidden_size, hidden_size), # 逆向きのEncoderの中間ベクトルを隠れ層サイズのベクトルに変換する線形結合層
            hh = L.Linear(hidden_size, hidden_size), # Decoderの中間ベクトルを隠れ層サイズのベクトルに変換する線形結合層
            hw = L.Linear(hidden_size, 1), # 隠れ層サイズのベクトルをスカラーに変換するための線形結合層
        )
        self.hidden_size = hidden_size # 隠れ層のサイズを記憶

        if device > -1: # numpyかcuda.cupyか
            self.xp = cuda.cupy
        else:
            self.xp = np

    def __call__(self, fs, bs, h):
        xp = self.xp
        # Attentionの計算
        # :param fs: 順向きのEncoderの中間ベクトルが記録されたリスト
        # :param bs: 逆向きのEncoderの中間ベクトルが記録されたリスト
        # :param h: Decoderで出力された中間ベクトル
        # :return: 順向きのEncoderの中間ベクトルの加重平均と逆向きのEncoderの中間ベクトルの加重平均
        batch_size = h.data.shape[0] # ミニバッチのサイズを記憶
        ws = [] # ウェイトを記録するためのリストの初期化
        sum_w = Variable(xp.zeros((batch_size, 1), dtype='float32')) # ウェイトの合計値を計算するための値を初期化
        # Encoderの中間ベクトルとDecoderの中間ベクトルを使ってウェイトの計算
        for f, b in zip(fs, bs):
            w = F.tanh(self.fh(f)+self.bh(b)+self.hh(h)) # 順向きEncoderの中間ベクトル、逆向きEncoderの中間ベクトル、Decoderの中間ベクトルを使ってウェイトの計算
            w = F.exp(self.hw(w)) # softmax関数を使って正規化する
            ws.append(w) # 計算したウェイトを記録
            sum_w += w
        # 出力する加重平均ベクトルの初期化
        att_f = Variable(xp.zeros((batch_size, self.hidden_size), dtype='float32'))
        att_b = Variable(xp.zeros((batch_size, self.hidden_size), dtype='float32'))
        for f, b, w in zip(fs, bs, ws):
            w /= sum_w # ウェイトの和が1になるように正規化
            # ウェイト * Encoderの中間ベクトルを出力するベクトルに足していく
            att_f += F.reshape(F.batch_matmul(f, w), (batch_size, self.hidden_size))
            att_b += F.reshape(F.batch_matmul(b, w), (batch_size, self.hidden_size))
        return att_f, att_b

# Attention Sequence to Sequence Modelクラス
class AttSeq2Seq(Chain):
    def __init__(self, input_size, hidden_size, batch_col_size, dropout, device):
        # Attention + Seq2Seqのインスタンス化
        # :param vocab_size: 語彙数のサイズ
        # :param embed_size: 単語ベクトルのサイズ
        # :param hidden_size: 隠れ層のサイズ
        super(AttSeq2Seq, self).__init__(
            f_encoder = LSTMEncoder(input_size, hidden_size), # 順向きのEncoder
            b_encoder = LSTMEncoder(input_size, hidden_size), # 逆向きのEncoder
            attention = Attention(hidden_size, device), # Attention Model
            decoder = AttLSTMDecoder(input_size, hidden_size) # Decoder
        )
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.decode_max_size = batch_col_size # デコードはEOSが出力されれば終了する、出力されない場合の最大出力語彙数
        self.dropout = dropout
        # 順向きのEncoderの中間ベクトル、逆向きのEncoderの中間ベクトルを保存するためのリストを初期化
        self.fs = []
        self.bs = []

        if device > -1: # numpyかcuda.cupyか
            xp = cuda.cupy
        else:
            xp = np
        self.xp = xp
        self.eos = xp.array([-3.449870e-01, -1.078110e-01,  2.550220e-01,  5.534310e-01,  -6.482100e-02,  7.935230e-01,  1.092545e+00,  1.060641e+00, 1.124500e-01, -6.563910e-01,  2.411870e-01, -3.978010e-01,
        8.743650e-01,  7.087400e-02,  3.342990e-01, -1.047685e+00, 6.116810e-01,  6.957860e-01,  8.381170e-01, -6.578200e-02, 4.172900e-01,  1.671147e+00, -4.299310e-01, -7.260790e-01, -1.051731e+00,  2.862650e-01, -2.223440e-01,  2.334280e-01,
       -4.886680e-01, -1.749300e-02, -2.235280e-01,  4.922340e-01, -6.045120e-01, -1.279066e+00,  6.746790e-01,  7.234180e-01, 4.587410e-01,  8.586320e-01,  8.506620e-01, -1.067123e+00,
       -4.976140e-01,  1.807070e-01, -2.823640e-01,  1.985520e-01, 1.871320e-01,  1.678450e-01, -8.955300e-01, -7.674100e-01,-7.150890e-01, -5.684570e-01,  1.566090e-01, -3.938570e-01,
       -8.346010e-01,  7.453450e-01, -6.536500e-01, -6.593530e-01, 7.730020e-01, -3.133780e-01, -5.066700e-02,  7.138140e-01,5.851100e-02, -1.188050e-01, -7.803950e-01,  5.129700e-01,
       -5.165130e-01, -2.536160e-01, -3.062480e-01, -7.261100e-01, 6.454010e-01, -6.184560e-01, -6.262320e-01,  1.214352e+00,6.061410e-01, -2.871380e-01,  6.147780e-01,  1.144356e+00,
       -4.542370e-01,  5.648800e-02, -6.982950e-01,  1.657411e+00, -8.480000e-03,  2.077230e-01, -4.453190e-01,  1.080613e+00,-5.794190e-01,  3.730560e-01, -1.074128e+00,  2.236350e-01,
       -7.794400e-01, -1.608600e-02, -8.614540e-01, -5.621870e-01,4.263780e-01,  1.937640e-01,  1.068030e-01, -1.394016e+00,-5.699780e-01,  8.450470e-01,  7.942400e-01,  3.378190e-01,
        2.659780e-01, -8.914890e-01,  1.229774e+00, -1.478740e-01,-3.944710e-01,  6.072830e-01, -7.140270e-01,  1.298476e+00,-9.445330e-01,  4.844720e-01, -6.218760e-01,  1.740110e-01,
       -8.913910e-01, -4.642170e-01, -1.607780e-01,  3.191190e-01,-1.508881e+00, -6.840500e-01,  3.966110e-01,  3.708550e-01, 4.208140e-01, -1.043645e+00,  2.296000e-03, -7.948100e-01,
        2.444190e-01, -2.037080e-01, -2.669670e-01,  3.852600e-02,-4.065940e-01, -9.121900e-01,  7.785260e-01,  4.747300e-02,6.296500e-02, -3.406880e-01, -6.997610e-01, -1.507166e+00,
       -3.125030e-01,  2.330780e-01, -5.722320e-01, -2.157140e-01, 5.087760e-01, -1.107566e+00, -2.554660e-01, -1.145372e+00,1.114480e-01,  3.536460e-01,  1.171659e+00, -6.270940e-01,
        9.305160e-01, -3.299710e-01,  2.693920e-01,  1.543254e+00, -1.637780e-01, -4.804100e-01,  6.141180e-01,  2.210130e-01,5.328000e-02,  8.703150e-01,  3.903550e-01, -1.324310e-01,
       -1.944590e-01,  7.707950e-01,  3.311660e-01,  3.853970e-01,5.341250e-01,  1.930890e-01,  2.329830e-01,  8.466800e-02,-1.184500e-02, -8.840580e-01, -1.267377e+00, -4.732300e-01,
       -2.192820e-01,  3.817530e-01,  1.100000e-04,  1.230055e+00, 5.427980e-01,  9.451940e-01, -5.206950e-01, -4.775500e-02,5.412130e-01,  7.279280e-01,  2.029600e-02,  8.678150e-01,
        1.126820e-01,  7.572610e-01,  8.657090e-01, -8.650310e-01,8.928000e-01, -6.137420e-01,  1.362390e-01, -5.708310e-01,6.162060e-01, -7.639380e-01, -5.017350e-01,  6.531750e-01,
       -2.253800e-02,  5.040320e-01, -8.807870e-01,  4.289790e-01], dtype=xp.float32)

    def encode(self, words, batch_size):
        # Encoderの計算
        # :param words: 入力で使用する単語記録されたリスト
        # :param batch_size: ミニバッチのサイズ
        # :return:
        # 内部メモリ、中間ベクトルの初期化
        xp = self.xp
        c = Variable(xp.zeros((batch_size, self.hidden_size), dtype='float32'))
        h = Variable(xp.zeros((batch_size, self.hidden_size), dtype='float32'))
        # 順向きのEncoderの計算
        for w in words:
            c, h = self.f_encoder(w, c, h)
            h = F.dropout(h, ratio=self.dropout)
            self.fs.append(h) # 計算された中間ベクトルを記録
        # 内部メモリ、中間ベクトルの初期化
        c = Variable(xp.zeros((batch_size, self.hidden_size), dtype='float32'))
        h = Variable(xp.zeros((batch_size, self.hidden_size), dtype='float32'))
        # 逆向きのEncoderの計算
        for w in reversed(words):
            c, h = self.b_encoder(w, c, h)
            h = F.dropout(h, ratio=self.dropout)
            self.bs.insert(0, h) # 計算された中間ベクトルを記録
        # 内部メモリ、中間ベクトルの初期化
        self.c = Variable(xp.zeros((batch_size, self.hidden_size), dtype='float32'))
        self.h = Variable(xp.zeros((batch_size, self.hidden_size), dtype='float32'))

    def decode(self, w):
        # Decoderの計算
        # :param w: Decoderで入力する単語
        # :return: 予測単語
        att_f, att_b = self.attention(self.fs, self.bs, self.h)
        t, self.c, self.h = self.decoder(w, self.c, self.h, att_f, att_b)
        return t

    def reset(self):
        # インスタンス変数を初期化する
        # Encoderの中間ベクトルを記録するリストの初期化
        self.fs = []
        self.bs = []
        # 勾配の初期化
        self.zerograds()

    def __call__(self, enc_words, dec_words=None, train=True):
        # 順伝播の計算を行う関数
        # :param enc_words: 発話文の単語を記録したリスト
        # :param dec_words: 応答文の単語を記録したリスト
        # :param train: 学習か予測か
        # :return: 計算した損失の合計 or 予測したデコード文字列
        enc_words = enc_words.transpose(1,0,2)[::-1] #逆にする！
        xp = self.xp
        if train:
            dec_words = dec_words.transpose(1,0,2)
        batch_size = len(enc_words[0]) # バッチサイズを記録
        self.reset() # model内に保存されている勾配をリセット
        enc_words = [Variable(xp.array(row, dtype='float32')) for row in enc_words] # 発話リスト内の単語をVariable型に変更
        self.encode(enc_words, batch_size) # エンコードの計算
        t = Variable(xp.array([self.eos for _ in range(batch_size)], dtype='float32'))
        loss = Variable(xp.zeros((), dtype='float32')) # 損失の初期化
        # デコーダーの計算
        if train: # 学習の場合は損失を計算する
            for w in dec_words:
                y = self.decode(t) # 1単語ずつをデコードする
                t = Variable(xp.array(w, dtype='float32')) # 正解単語をVariable型に変換
                loss += F.mean_squared_error(y, t) # 正解単語と予測単語を照らし合わせて損失を計算
            return loss
        else: # 予測の場合はデコード文字列を生成する
            ys = [] # デコーダーが生成する単語を記録するリスト
            for i in range(self.decode_max_size):
                y = self.decode(t)
                ys.append(y.data[0])
                t = y
                # if y == self.eos: # EOSを出力したならばデコードを終了する
                #     break
            return ys

