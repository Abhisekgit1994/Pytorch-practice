# LeNet Architecture

![lenet-5](https://user-images.githubusercontent.com/55094650/225531696-edae9e8c-fe90-40e4-95a8-e950417fed0e.png)

#### LeNet implementation using PyTorch 
1. 6 * Convolution ( 5,5), padding = 0, stride =1 and ReLU           torch.Size([64, 6, 28, 28])
2. AvgPooling kernel (5,5), stride = 2,2                         torch.Size([64, 6, 14, 14])
3. 16 * Convolution ( 5,5), padding = 0, stride =1 and ReLU           torch.Size([64, 16, 10, 10])
4. AvgPooling kernel (5,5), stride = 2,2                        torch.Size([64, 16, 5, 5])
5. 120 * Convolution ( 5,5), padding = 0, stride =1 and ReLU           torch.Size([64, 120, 1, 1])
6. Linear                                                        torch.Size([64, 84])
7. Linear                                                        torch.Size([64, 10])

#### Model progress and Performance
# ****** training the model********
→ Epoch [1/3], Step [64/938], Loss: 0.8119 <br>
→ Epoch [1/3], Step [128/938], Loss: 0.5270 <br>
→ Epoch [1/3], Step [192/938], Loss: 0.2594 <br>
→ Epoch [1/3], Step [256/938], Loss: 0.2946 <br>...<br>
→ Epoch [3/3], Step [448/938], Loss: 0.0230 <br>
→ Epoch [3/3], Step [512/938], Loss: 0.0226 <br>
→ Epoch [3/3], Step [576/938], Loss: 0.0943 <br>
→ Epoch [3/3], Step [640/938], Loss: 0.0605 <br>
→ Epoch [3/3], Step [704/938], Loss: 0.0403 <br>
→ Epoch [3/3], Step [768/938], Loss: 0.0160 <br>
→ Epoch [3/3], Step [832/938], Loss: 0.0340 <br>
→ Epoch [3/3], Step [896/938], Loss: 0.0877 <br>
→ Epoch [3/3], Step [938/938], Loss: 0.0203 <br>
→ Loss at epoch 3 0.05773831920776226 <br>
→ ******** Printing accuracy for LeNet model *********** <br>
→ checking accuracy on train data <br>
→ Accuracy is 98.8550033569336 <br>
→ checking accuracy on test data <br>
→ Accuracy is 98.6199951171875 <br>

###### Thankyou for visiting my profile