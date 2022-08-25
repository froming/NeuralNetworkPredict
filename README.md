# NeuralNetworkPredict
<1>DESCRIPTION
The project use django and neuralnetwork to predict strain at break,stress at break,tan deltaThe progect is designed by c/s.you can write csv file and set filename of model on train html.When you train a model after five minutes,you can predict strain at break,stress at break,tan delta in predict html.Csv file'format demand input variables of twelve,they are CO_W NH_Aper NH_W HS% SS% %aromati %cyclic CED solubility CORE_pctWgt sol_pctWgt Mtw,also output varables of output Strain at break,stress at break,tan delta.




<2>ILLUSTRATE
1)Trained model file in train/upload.
2)Uploaded csv is used,they are deleted.Before deleted,thay are saved in train/upload
3)The code enable set default model and csv file.You can set them in train/.
4)The code enable set introduce csv.You can save the file in static/Predictimage after writed.
5)The code enable train'image you can install training image,also find they in static/Predictimage.
6)The code enable train multiple format model,but just predict Strain at break,stress at break,tan delta,if you trained model,you can find model like 'XXXX.pt' in train/upload.
