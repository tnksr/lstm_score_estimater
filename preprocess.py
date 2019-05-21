# coding: utf-8
import re
import unicodedata
import MeCab


class Preprocess(object):
    def __init__(self, dictionary_path, valid_posid, stop_words=None):
        # wakati
        self.mecab = MeCab.Tagger('-Odump -d ' + dictionary_path)
        self.valid_posid = valid_posid
        self.stop_words = stop_words
        # normalize
        self.regexs = {
            'html': re.compile(r'<[^>]+>|&quot;'),
            'url': re.compile(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-]+'),
            'sign': re.compile(r'[\W]', re.U),
            'space': re.compile(r'([  \t\n\r\f\v\s]|\\n|\\t|\\r|\\f|\\v)+', re.U),
            'vowel': re.compile(r'[ー]+'),
            'alphabet': re.compile(r'[a-zA-Z]', re.U),
            'digit': re.compile(r'[0-9\d]+'),
            'numeric': re.compile(r'([¥￥]{1}[0-9]+[十百千万]*|[0-9]*[0-9]+(,[0-9]{3}|:[0-9]{2}|\/[0-9]{1,2}|-[0-9]*))')
        }


    def __call__(self, string):
        return self.wakati(self.cleansing(string))        

    def cleansing(self, string):
        """
        正規化するメソッド

        Args :
            string : 元データ
        Returns :
            string : 正規化後のテキストデータ

        """
        # 全角英数字を半角にする
        string = unicodedata.normalize('NFKC', string)
        # スペースを半角スペースに統一する
        string = re.sub(self.regexs['space'], ' ', string.strip())
        # HTML, URLの削除
        string = re.sub(self.regexs['html'], ' ', string)
        string = re.sub(self.regexs['url'], ' ', string)
        # 数字の削除
        string = re.sub(self.regexs['numeric'], ' ', string)
        # 記号の削除
        string = re.sub(self.regexs['sign'], ' ', string)
        # 長音文字を1つにする
        string = re.sub(self.regexs['vowel'], u'ー', string)
        # 英字間以外のスペースを削除
        tokens = string.split()
        string = ''
        if len(tokens) >= 1:
            for i in range(len(tokens)-1):
                pre = tokens[i][-1]
                behind = tokens[i+1][0]
                if re.match(self.regexs['alphabet'], pre) and re.match(self.regexs['alphabet'], behind):
                    string += tokens[i] + ' '
                else:
                    string += tokens[i]
            string += tokens[len(tokens)-1]
        return string

    def wakati(self, string):
        """
        分かち書きをするメソッド

        Args :
            string : 正規化後のテキストデータ
        Returns :
            wakati : 分かち書き後の単語を要素とする配列

        """
        wakati = []
        for token in self.mecab.parse(string).split('\n')[1:-2]:
            morpheme = token.split()[1].lower()
            posid = token.split()[7]
            if posid in self.valid_posid and not re.match(self.regexs['digit'], morpheme):
                if self.stop_words and morpheme in self.stop_words: continue
                wakati.append(morpheme)
        return wakati

if __name__ == '__main__':
    print ('preprocess.py')
