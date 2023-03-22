import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from custom_text_data import data_loader
from utils import save_checkpoint, load_checkpoint, print_examples
from model import EncoderToDecoder

torch.backends.cudnn.benchmark = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
    # transforms.RandomCrop((356, 356)),
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# load dataset
train_loader, dataset = data_loader(root_dir='../../flickr8k/Images', caption_file='../../flickr8k/captions.txt',
                                    transform=transform,
                                    num_workers=2,
                                    batch_size=32
                                    )

# hyper parameters
embed_size = 256
hidden_size = 256
vocab_size = len(dataset.vocab)
num_layers = 2
lr = 4e-4
epochs = 100

# tensorboard
writer = SummaryWriter('progress')

# load model
model = EncoderToDecoder(embed_size, hidden_size, vocab_size, num_layers).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.strtoid['<PAD>'])
optimizer = optim.Adam(model.parameters(), lr=lr)


def train_model(step=0):
    model.train()

    for epoch in range(epochs):
        losses = []
        for idx, (imgs, caps) in enumerate(train_loader):
            imgs = imgs.to(device)
            captions = caps.to(device)
            # print(captions.shape)

            outputs = model(imgs, captions[:-1])
            # print(outputs)
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))

            writer.add_scalar('Training loss', loss.item(), global_step=step)
            step += 1

            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()
            losses.append(loss.item())

            if (idx + 1) % 32 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}], "
                    f"Step [{idx + 1}/{len(train_loader)}], "
                    f"Loss: {loss.item():.4f}, "

                )
            elif (idx + 1) == len(train_loader):
                print(
                    f"Epoch [{epoch + 1}/{epochs}], "
                    f"Step [{idx + 1}/{len(train_loader)}], "
                    f"Loss: {loss.item():.4f}"
                )

        print("Loss at epoch", epoch + 1, sum(losses) / len(losses))


if __name__ == '__main__':
    # print(len(train_loader))
    train_model()
