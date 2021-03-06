FROM ubuntu:16.04

MAINTAINER Peinan

# 必要なパッケージインストール
RUN apt-get update -y \
    && apt-get install -y wget build-essential gcc zlib1g-dev libssl-dev locales \
                          tk-dev libgdbm-dev libc6-dev libbz2-dev git sudo vim \
                          libreadline-gplv2-dev libncursesw5-dev libsqlite3-dev \
                          make curl xz-utils file mecab libmecab-dev mecab-ipadic-utf8 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 文字化け対策の locale 設定
RUN echo "ja_JP UTF-8" > /etc/locale.gen && locale-gen
ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL ja_JP.UTF-8

WORKDIR /root/

# neologd のインストール
RUN git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git
WORKDIR /root/mecab-ipadic-neologd
RUN ./bin/install-mecab-ipadic-neologd -n -y

WORKDIR /root

# python3.6 のインストール
RUN wget https://www.python.org/ftp/python/3.6.6/Python-3.6.6.tgz \
    && tar zxf Python-3.6.6.tgz \
    && cd Python-3.6.6 \
    && ./configure \
    && make altinstall \
    && ln -s /usr/local/bin/python3.6 /usr/local/bin/python \
    && ln -s /usr/local/bin/pip3.6 /usr/local/bin/pip

ENV PYTHONIOENCODING "utf-8"

# 必要な pip パッケージインストール
COPY requirements.txt $HOME
RUN pip install -U pip \
    && pip install -r requirements.txt

COPY app $HOME/app
EXPOSE 80

WORKDIR $HOME/app
CMD gunicorn -w 1 --timeout 300 -b 0.0.0.0:80 app:app
