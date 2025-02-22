from models.mnist_model import MNISTModel
from utils.data_loader import load_data
from utils.trainer import train_model, eval_model
import argparse

def main(resume_training):
    model = MNISTModel(resume_training=resume_training)
    train_loader, eval_loader = load_data(3000) # batch_size = 3000

    EPOCHS = 5
    train_model(model, EPOCHS, train_loader)
    eval_model(model, eval_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='Resume training from saved weights')
    parser.add_argument('--scratch', action='store_true', help='Start training from scratch')
    args = parser.parse_args()

    if not (args.resume or args.scratch):
        parser.error("Aucun argument fourni. Veuillez spécifier --resume ou --scratch.")

    if args.resume:
        main(True)
    elif args.scratch:
        main(False)

