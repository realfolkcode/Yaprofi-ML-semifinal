FROM continuumio/anaconda3:latest

RUN conda config --add channels conda-forge
RUN conda install catboost
RUN conda install networkx

COPY solve.py solve.py
COPY tmp tmp
COPY model_group_reg_cv model_group_reg_cv

CMD mkdir /var/log
CMD python solve.py /tmp/data/test.tsv > /var/log/result