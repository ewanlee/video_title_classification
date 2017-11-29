This project is developed under python2, Ubuntu Server 16.04 and I use one
Nvidia GPX 1080 Ti graphic card.

Notice that the word2vec model files are not uploaded to this repository because of
Github upload size limitation.

You can find the model files at [here](https://pan.baidu.com/s/1o89R8Oa).

I have used three models:
- TextCNN
- TextRNN
- TextRCNN

Model define files are as follows:
- textcnn.py
- textrnn.py
- textrcnn.py

Model training files are as follows:
- train_cnn.py
- train_rnn.py
- train_rcnn.py

You can view more details on comments in source code.

For now, I get the best multi-class classification accuracy by use TextCNN. The
hyperparameters have written in code.
