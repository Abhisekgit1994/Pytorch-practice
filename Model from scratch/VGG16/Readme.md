
# => VGG16 Architecture
# => VGG16 Layers
### VGG16 implementation from scratch with PyTorch :

→ 2 x Convolution layer of 64 channel of 3x3 kernel and same padding

→ 1 x Maxpooling layer of 2x2 pool size and stride 2x2

→ 2 x convolution layer of 128 channel of 3x3 kernel and same padding

→ 1 x Maxpooling layer of 2x2 pool size and stride 2x2

→ 3 x convolution layer of 256 channel of 3x3 kernel and same padding

→ 1 x Maxpooling layer of 2x2 pool size and stride 2x2

→ 3 x convolution layer of 512 channel of 3x3 kernel and same padding

→ 1 x Maxpooling layer of 2x2 pool size and stride 2x2

→ 3 x Convolution layer of 512 channel of 3x3 kernel and same padding

→ 1 x Maxpooling layer of 2x2 pool size and stride 2x2

→ 3 x Linear Layers

#### ***** Model progress and Performance on CIFAR10 dataset : 1000 classes *****
→ Epoch [1/1], Step [8/6250], Loss: 26.3157, iteration: 1, time: 0.11(s),<br>
→ Epoch [1/1], Step [16/6250], Loss: 10.3575, iteration: 2, time: 0.11(s),<br>
→ Epoch [1/1], Step [24/6250], Loss: 7.4232, iteration: 3, time: 0.11(s),<br>
→ Epoch [1/1], Step [32/6250], Loss: 2.1376, iteration: 4, time: 0.11(s),<br>....<br>
→ Epoch [1/1], Step [6224/6250], Loss: 1.5443, iteration: 778, time: 0.11(s),<br>
→ Epoch [1/1], Step [6232/6250], Loss: 2.0813, iteration: 779, time: 0.11(s),<br>
→ Epoch [1/1], Step [6240/6250], Loss: 2.0809, iteration: 780, time: 0.11(s),<br>
→ Epoch [1/1], Step [6248/6250], Loss: 2.1535, iteration: 781, time: 0.11(s),<br>
→ Epoch [1/1], Step [6250/6250], Loss: 2.1855iteration: 781, time: 0.11(s),<br>
→ Loss at epoch 1 2.198171835727692<br>
#### ******** Printing accuracy for VGG16 model ***********<br>
→ checking accuracy on test data<br>
→ Accuracy is 30.059999465942383<br>

#### Accuracy is 30% after training for 1 epoch only.

######  Thankyou for visiting my profile