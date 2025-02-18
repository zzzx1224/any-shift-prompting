from __future__ import print_function


import yaml
import os, shutil
import argparse
import time
from tensorboardX import SummaryWriter
from aug import *
import pdb
from opendata import *
from dataset import *
import model
import basemodel
import com_model
import sys
from tqdm import tqdm

parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
parser.add_argument("--lr", default=0.0001, type=float, help="learning rate")
parser.add_argument("--weight_decay", default=5e-4, type=float, help="learning rate")
parser.add_argument("--sparse", default=0, type=float, help="L1 panelty")
parser.add_argument(
    "--resume", "-r", action="store_true", help="resume from checkpoint"
)
parser.add_argument("--gpu", default="0", help="GPU to use [default: GPU 0]")
parser.add_argument("--log_dir", default="log1", help="Log dir [default: log]")
parser.add_argument("--dataset", default="PACS", help="datasets")
parser.add_argument(
    "--batch_size",
    type=int,
    default=128,
    help="Batch Size during training [default: 32]",
)
parser.add_argument(
    "--bases", type=int, default=7, help="Batch Size during training [default: 32]"
)
parser.add_argument(
    "--shuffle", type=int, default=0, help="Batch Size during training [default: 32]"
)
parser.add_argument(
    "--optimizer", default="adam", help="adam or momentum [default: adam]"
)
parser.add_argument("--sharing", default="layer", help="Log dir [default: log]")
parser.add_argument("--net", default="clip", help="res18 or res50 or clip")
parser.add_argument("--l2", action="store_true")
parser.add_argument("--base", action="store_true")
parser.add_argument("--autodecay", action="store_true")
parser.add_argument("--share_bases", action="store_true")
parser.add_argument("--hychy", type=int, default=0, help="hyrarchi")
parser.add_argument("--sub", default=1.0, type=float, help="subset of tinyimagenet")
parser.add_argument(
    "--test_domain", default="sketch", help="GPU to use [default: GPU 0]"
)
parser.add_argument("--train_domain", default="", help="GPU to use [default: GPU 0]")
parser.add_argument("--ite_train", default=True, type=bool, help="iterative training")
parser.add_argument("--max_ite", default=3000, type=int, help="maximal iteration")
parser.add_argument("--test_ite", default=50, type=int, help="test iteration")
parser.add_argument("--bias", default=1, type=int, help="whether to use bias")
parser.add_argument("--test_batch", default=32, type=int, help="test batch size")
parser.add_argument("--data_aug", default=1, type=int, help="data augmentation")
parser.add_argument("--difflr", default=0, type=int, help="different learning rates")
parser.add_argument(
    "--mctime", default=1, type=int, help="number of Monte Carlo samples"
)
parser.add_argument("--cbeta", default=1, type=float, help="beta for common kl")
parser.add_argument("--vptmethod", default="cat", help="concat vision and text features to generate prompts")
parser.add_argument(
    "--test_sample", default=False, type=bool, help="sampling in test time"
)
parser.add_argument("--doms_pro", default=0, type=int, help="use domain names in prompt")
parser.add_argument("--learnpro", default=0, type=int, help="learnable prompt")
parser.add_argument("--featpro", default=1, type=int, help="use image features in prompt")
parser.add_argument("--vpro", default=1, type=int, help="sample prompt from the distribution")
parser.add_argument("--imgp", default=0, type=int, help="only image prompts")
parser.add_argument("--taskp", default=1, type=int, help="task prompts")
parser.add_argument("--taskpres", default=1, type=int, help="residual calculation on task prompts")
parser.add_argument("--opendg", default=0, type=int, help="open domain generalization")
parser.add_argument("--opentra", default="all", help="all or batch")
parser.add_argument("--trainpro", default=1, type=int, help="train the prompt")
parser.add_argument("--reslr", default=1, type=float, help="learning rate coefficient for visual prompt branch")
parser.add_argument(
    "--cfgfile", default="pacs_config.yaml", help="config file"
)
parser.add_argument("--seed", default=42, type=int, help="seed")
parser.add_argument("--transl", default=2, type=int, help="transformer layers")


args = parser.parse_args()
BATCH_SIZE = args.batch_size
OPTIMIZER = args.optimizer
gpu_index = args.gpu
backbone = args.net
max_ite = args.max_ite
test_ite = args.test_ite
test_batch = args.test_batch
transl = args.transl
iteration_training = args.ite_train
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

test_domain = args.test_domain
train_domain = args.train_domain
# a_beta = args.abeta
# p_beta = args.pbeta
ifsample = args.test_sample
difflr = args.difflr
res_lr = args.reslr
c_beta = args.cbeta
# agg_model = args.agg_model
# agg_method = args.agg_method
# prior_test = args.ptest
# prior_test = bool(prior_test)
with_bias = args.bias
with_bias = bool(with_bias)
difflr = bool(difflr)
# random_domain = bool(args.randdom)
# sharemodel = bool(args.sharem)

doms_pro = args.doms_pro
learnpro = args.learnpro
featpro = args.featpro
trainpro = args.trainpro
vpro = args.vpro
mc_times = args.mctime
# ifcommon = args.ifcommon
# ifadapt = args.ifadapt

data_aug = args.data_aug
data_aug = bool(data_aug)

LOG_DIR = os.path.join("logs", args.log_dir)
args.log_dir = LOG_DIR

name_file = sys.argv[0]
if os.path.exists(LOG_DIR):
    shutil.rmtree(LOG_DIR)
os.mkdir(LOG_DIR)
os.mkdir(LOG_DIR + "/train_img")
os.mkdir(LOG_DIR + "/test_img")
os.mkdir(LOG_DIR + "/files")
os.system("cp %s %s" % (name_file, LOG_DIR))
os.system("cp %s %s" % ("*.py", os.path.join(LOG_DIR, "files")))
os.system("cp -r %s %s" % ("models", os.path.join(LOG_DIR, "files")))
LOG_FOUT = open(os.path.join(LOG_DIR, "log_train.txt"), "w")
print(args)
LOG_FOUT.write(str(args) + "\n")


def log_string(out_str, print_out=True):
    LOG_FOUT.write(out_str + "\n")
    LOG_FOUT.flush()
    if print_out:
        print(out_str)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_random_seed(args.seed)

st = " "
log_string(st.join(sys.argv))


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


device = "cuda" if torch.cuda.is_available() else "cpu"
best_acc = 0  # best test accuracy
best_accp = 0
best_valid_acc = 0  # best validation accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


writer = SummaryWriter(log_dir=args.log_dir)

# Data
print("==> Preparing data..")

bird = False

decay_inter = [250, 450]

if args.dataset == "PACS":
    NUM_CLASS = 7
    num_domain = 4
    batchs_per_epoch = 0
    domains = ["art_painting", "photo", "cartoon", "sketch"]
    classnames = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
    if args.opendg:
        train_classnames = classnames[:6]
    else:
        train_classnames = classnames
    # pdb.set_trace()
    assert test_domain in domains
    domains.remove(test_domain)
    if train_domain:
        domains = train_domain.split(",")
    log_string("data augmentation is " + str(data_aug))
    if data_aug:
        # log_string()
        transform_train = transforms.Compose(
            [
                # transforms.RandomCrop(64, padding=4),
                transforms.RandomResizedCrop(
                    224, scale=(0.8, 1.2), ratio=(0.75, 1.33), interpolation=2
                ),
                transforms.RandomHorizontalFlip(),
                ImageJitter(jitter_param),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    else:
        transform_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    log_string("train_domain: " + str(domains))
    log_string("test: " + str(test_domain))

    # if not random_domain:
    if args.opendg:
        all_dataset = opPACS(test_domain)
    else:
        all_dataset = PACS(test_domain)
    #     all_dataset = randPACS(test_domain)

elif args.dataset == "office":
    NUM_CLASS = 65
    num_domain = 4
    batchs_per_epoch = 0
    # ctx_test = 10
    domains = ["art", "clipart", "product", "real_World"]
    classnames = [
        "Alarm_Clock",
        "Backpack",
        "Batteries",
        "Bed",
        "Bike",
        "Bottle",
        "Bucket",
        "Calculator",
        "Calendar",
        "Candles",
        "Chair",
        "Clipboards",
        "Computer",
        "Couch",
        "Curtains",
        "Desk_Lamp",
        "Drill",
        "Eraser",
        "Exit_Sign",
        "Fan",
        "File_Cabinet",
        "Flipflops",
        "Flowers",
        "Folder",
        "Fork",
        "Glasses",
        "Hammer",
        "Helmet",
        "Kettle",
        "Keyboard",
        "Knives",
        "Lamp_Shade",
        "Laptop",
        "Marker",
        "Monitor",
        "Mop",
        "Mouse",
        "Mug",
        "Notebook",
        "Oven",
        "Pan",
        "Paper_Clip",
        "Pen",
        "Pencil",
        "Postit_Notes",
        "Printer",
        "Push_Pin",
        "Radio",
        "Refrigerator",
        "Ruler",
        "Scissors",
        "Screwdriver",
        "Shelf",
        "Sink",
        "Sneakers",
        "Soda",
        "Speaker",
        "Spoon",
        "TV",
        "Table",
        "Telephone",
        "ToothBrush",
        "Toys",
        "Trash_Can",
        "Webcam",
    ]
    if args.opendg:
        train_classnames = classnames[:54]
    else:
        train_classnames = classnames
    assert test_domain in domains
    domains.remove(test_domain)
    if train_domain:
        domains = train_domain.split(",")
    log_string("data augmentation is " + str(data_aug))
    if data_aug:
        # log_string()
        transform_train = transforms.Compose(
            [
                # transforms.RandomCrop(64, padding=4),
                transforms.RandomResizedCrop(
                    224, scale=(0.8, 1.2), ratio=(0.75, 1.33), interpolation=2
                ),
                transforms.RandomHorizontalFlip(),
                ImageJitter(jitter_param),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    else:
        transform_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    log_string("train_domain: " + str(domains))
    log_string("test: " + str(test_domain))

    if args.opendg:
        all_dataset = opOfficeHome(test_domain)
    else:
        all_dataset = OfficeHome(test_domain)

elif args.dataset == "vlcs":
    NUM_CLASS = 5
    num_domain = 4
    batchs_per_epoch = 0
    domains = ["CALTECH", "LABELME", "PASCAL", "SUN"]
    classnames = ["bird", "car", "chair", "dog", "person"]
    assert test_domain in domains
    domains.remove(test_domain)
    if train_domain:
        domains = train_domain.split(",")
    log_string("data augmentation is " + str(data_aug))
    if args.opendg:
        train_classnames = classnames[:4]
    else:
        train_classnames = classnames
    if data_aug:
        # log_string()
        transform_train = transforms.Compose(
            [
                # transforms.RandomCrop(64, padding=4),
                transforms.RandomResizedCrop(
                    224, scale=(0.8, 1.2), ratio=(0.75, 1.33), interpolation=2
                ),
                transforms.RandomHorizontalFlip(),
                ImageJitter(jitter_param),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    else:
        transform_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    log_string("train_domain: " + str(domains))
    log_string("test: " + str(test_domain))

    # pdb.set_trace()

    # if not random_domain:
    if args.opendg:
        all_dataset = opVLCS(test_domain)
    else:
        all_dataset = VLCS(test_domain)
    # else:
    #     all_dataset = randPACS(test_domain)

elif args.dataset == "domainnet":
    NUM_CLASS = 345
    num_domain = 6
    batchs_per_epoch = 0
    # ctx_test = 10
    domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
    classnames = ['aircraft_carrier', 'airplane', 'alarm_clock', 'ambulance', 'angel', 'animal_migration', 'ant', 'anvil', 'apple', 'arm', 'asparagus', 'axe', 'backpack', 'banana', 'bandage', 'barn', 'baseball', 'baseball_bat', 'basket', 'basketball', 'bat', 'bathtub', 'beach', 'bear', 'beard', 'bed', 'bee', 'belt', 'bench', 'bicycle', 'binoculars', 'bird', 'birthday_cake', 'blackberry', 'blueberry', 'book', 'boomerang', 'bottlecap', 'bowtie', 'bracelet', 'brain', 'bread', 'bridge', 'broccoli', 'broom', 'bucket', 'bulldozer', 'bus', 'bush', 'butterfly', 'cactus', 'cake', 'calculator', 'calendar', 'camel', 'camera', 'camouflage', 'campfire', 'candle', 'cannon', 'canoe', 'car', 'carrot', 'castle', 'cat', 'ceiling_fan', 'cello', 'cell_phone', 'chair', 'chandelier', 'church', 'circle', 'clarinet', 'clock', 'cloud', 'coffee_cup', 'compass', 'computer', 'cookie', 'cooler', 'couch', 'cow', 'crab', 'crayon', 'crocodile', 'crown', 'cruise_ship', 'cup', 'diamond', 'dishwasher', 'diving_board', 'dog', 'dolphin', 'donut', 'door', 'dragon', 'dresser', 'drill', 'drums', 'duck', 'dumbbell', 'ear', 'elbow', 'elephant', 'envelope', 'eraser', 'eye', 'eyeglasses', 'face', 'fan', 'feather', 'fence', 'finger', 'fire_hydrant', 'fireplace', 'firetruck', 'fish', 'flamingo', 'flashlight', 'flip_flops', 'floor_lamp', 'flower', 'flying_saucer', 'foot', 'fork', 'frog', 'frying_pan', 'garden', 'garden_hose', 'giraffe', 'goatee', 'golf_club', 'grapes', 'grass', 'guitar', 'hamburger', 'hammer', 'hand', 'harp', 'hat', 'headphones', 'hedgehog', 'helicopter', 'helmet', 'hexagon', 'hockey_puck', 'hockey_stick', 'horse', 'hospital', 'hot_air_balloon', 'hot_dog', 'hot_tub', 'hourglass', 'house', 'house_plant', 'hurricane', 'ice_cream', 'jacket', 'jail', 'kangaroo', 'key', 'keyboard', 'knee', 'knife', 'ladder', 'lantern', 'laptop', 'leaf', 'leg', 'light_bulb', 'lighter', 'lighthouse', 'lightning', 'line', 'lion', 'lipstick', 'lobster', 'lollipop', 'mailbox', 'map', 'marker', 'matches', 'megaphone', 'mermaid', 'microphone', 'microwave', 'monkey', 'moon', 'mosquito', 'motorbike', 'mountain', 'mouse', 'moustache', 'mouth', 'mug', 'mushroom', 'nail', 'necklace', 'nose', 'ocean', 'octagon', 'octopus', 'onion', 'oven', 'owl', 'paintbrush', 'paint_can', 'palm_tree', 'panda', 'pants', 'paper_clip', 'parachute', 'parrot', 'passport', 'peanut', 'pear', 'peas', 'pencil', 'penguin', 'piano', 'pickup_truck', 'picture_frame', 'pig', 'pillow', 'pineapple', 'pizza', 'pliers', 'police_car', 'pond', 'pool', 'popsicle', 'postcard', 'potato', 'power_outlet', 'purse', 'rabbit', 'raccoon', 'radio', 'rain', 'rainbow', 'rake', 'remote_control', 'rhinoceros', 'rifle', 'river', 'roller_coaster', 'rollerskates', 'sailboat', 'sandwich', 'saw', 'saxophone', 'school_bus', 'scissors', 'scorpion', 'screwdriver', 'sea_turtle', 'see_saw', 'shark', 'sheep', 'shoe', 'shorts', 'shovel', 'sink', 'skateboard', 'skull', 'skyscraper', 'sleeping_bag', 'smiley_face', 'snail', 'snake', 'snorkel', 'snowflake', 'snowman', 'soccer_ball', 'sock', 'speedboat', 'spider', 'spoon', 'spreadsheet', 'square', 'squiggle', 'squirrel', 'stairs', 'star', 'steak', 'stereo', 'stethoscope', 'stitches', 'stop_sign', 'stove', 'strawberry', 'streetlight', 'string_bean', 'submarine', 'suitcase', 'sun', 'swan', 'sweater', 'swing_set', 'sword', 'syringe', 'table', 'teapot', 'teddy-bear', 'telephone', 'television', 'tennis_racquet', 'tent', 'The_Eiffel_Tower', 'The_Great_Wall_of_China', 'The_Mona_Lisa', 'tiger', 'toaster', 'toe', 'toilet', 'tooth', 'toothbrush', 'toothpaste', 'tornado', 'tractor', 'traffic_light', 'train', 'tree', 'triangle', 'trombone', 'truck', 'trumpet', 't-shirt', 'umbrella', 'underwear', 'van', 'vase', 'violin', 'washing_machine', 'watermelon', 'waterslide', 'whale', 'wheel', 'windmill', 'wine_bottle', 'wine_glass', 'wristwatch', 'yoga', 'zebra', 'zigzag']
    if args.opendg:
        train_classnames = classnames[:300]
    else:
        train_classnames = classnames
    assert test_domain in domains
    domains.remove(test_domain)
    if train_domain:
        domains = train_domain.split(",")
    log_string("data augmentation is " + str(data_aug))
    if data_aug:
        # log_string()
        transform_train = transforms.Compose(
            [
                # transforms.RandomCrop(64, padding=4),
                transforms.RandomResizedCrop(
                    224, scale=(0.8, 1.2), ratio=(0.75, 1.33), interpolation=2
                ),
                transforms.RandomHorizontalFlip(),
                ImageJitter(jitter_param),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    else:
        transform_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    transform_test = transforms.Compose(
        [
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    log_string("train_domain: " + str(domains))
    log_string("test: " + str(test_domain))

    if args.opendg:
        all_dataset = opDomainNet(test_domain)
    else:
        all_dataset = DomainNet(test_domain)
    # rt_context = rtOF(test_domain, ctx_num)

else:
    raise NotImplementedError


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {"Total": total_num, "Trainable": trainable_num}


args.num_classes = NUM_CLASS
args.num_domains = num_domain
args.bird = bird


# Model
print("==> Building model..")
f = open(args.cfgfile)
cfg = yaml.safe_load(f)
# classnames = classnames
# pdb.set_trace()
print(f"Loading CLIP (backbone: {cfg['MODEL']['BACKBONE']['NAME']})")
clip_model = model.load_clip_to_cpu(cfg)
if (
    cfg["TRAINER"]["COCOOP"]["PREC"] == "fp32"
    or cfg["TRAINER"]["COCOOP"]["PREC"] == "amp"
):
    # CLIP's default precision is fp16
    clip_model.float()
# if args.dataset=='PACS' or args.dataset=='office':
#     net = model.CustomCLIP(cfg, classnames, clip_model)
print("Building custom CLIP")
imgencoder = None
if difflr:
    if backbone == "res18" or backbone == "res50":
        imgencoder = basemodel.encoder(backbone)
if trainpro:
    net = model.CustomCLIP(
        cfg,
        classnames,
        clip_model,
        doms_pro,
        test_domain,
        learnpro,
        vpro,
        args.opendg,
        args.dataset,
        imgencoder,
        args.imgp,
        args.taskp,
        args.taskpres,
        args.vptmethod,
        args.transl,
    )
else:
    net = com_model.CustomCLIP(
        cfg,
        classnames,
        clip_model,
        doms_pro,
        test_domain,
        learnpro,
        args.opendg,
        args.dataset,
        imgencoder,
        args.imgp,
        args.taskp,
        args.taskpres,
        args.vptmethod,
    )

print("Turning off gradients in both the image and the text encoder")
name_to_update = "prompt_learner"

# for name, param in net.named_parameters():
#     if name_to_update not in name:
#         param.requires_grad_(False)
net.text_encoder.requires_grad_(False)
if not difflr:
    net.image_encoder.requires_grad_(False)

# Double check
enabled = set()
for name, param in net.named_parameters():
    if param.requires_grad:
        enabled.add(name)
print(f"Parameters to be updated: {enabled}")

# if cfg.MODEL.INIT_WEIGHTS:
#     load_pretrained_weights(net.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

net.to(device)
# NOTE: only give prompt_learner to the optimizer
# self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
# self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
# self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

# self.scaler = GradScaler() if cfg.TRAINER.COCOOP.PREC == "amp" else None
# self.scaler = None

# Note that multi-gpu training could be slow because CLIP's size is
# big, which slows down the copy operation in DataParallel
device_count = torch.cuda.device_count()
# if device_count > 1:
#     print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
#     net = nn.DataParallel(net)
########################################################
log_string(str(net.extra_repr))

pc = get_parameter_number(net)
log_string(
    "Total: %.4fM, Trainable: %.4fM"
    % (pc["Total"] / float(1e6), pc["Trainable"] / float(1e6))
)

net.train()
if device == "cuda":
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# pdb.set_trace()
#
# for name, param in net.visualpromptlearner.named_parameters():
#     if param.requires_grad:
#         print(name)
#
# pdb.set_trace()
#
# for name, param in net.image_encoder.named_parameters():
#     if param.requires_grad:
#         print(name)
#
# pdb.set_trace()

if args.resume:
    # Load checkpoint.
    print("==> Resuming from checkpoint..")
    assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found!"
    checkpoint = torch.load("./checkpoint/ckpt.t7")
    net.load_state_dict(checkpoint["net"])
    best_acc = checkpoint["acc"]
    start_epoch = checkpoint["epoch"]

criterion = nn.CrossEntropyLoss()

WEIGHT_DECAY = args.weight_decay

if OPTIMIZER == "momentum":
    optimizer = torch.optim.SGD(
        net.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY, momentum=0.9
    )
elif OPTIMIZER == "nesterov":
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=args.lr,
        weight_decay=WEIGHT_DECAY,
        momentum=0.9,
        nesterov=True,
    )
########################################################
########################################################
########################################################
########################################################
elif OPTIMIZER == "adam":
    # pdb.set_trace()
    if not args.taskp:
        if args.imgp and (args.vptmethod == 'cat' or args.vptmethod == 'macat' or args.vptmethod == 'mcat'):
            log_string('both text and visual prompt learner')
            optimizer = torch.optim.Adam(
                [
                    {"params": net.prompt_learner.parameters()},
                    {"params": net.visualpromptlearner.parameters(), "lr": args.lr * res_lr},
                    {"params": net.claposembedding, "lr": args.lr * res_lr},
                    # {"params": net.visualpromptlearner.firstembedding},
                ],
                lr=args.lr,
                weight_decay=WEIGHT_DECAY,
            )
        elif args.imgp and (args.vptmethod != 'cat' and args.vptmethod != 'macat' and args.vptmethod != 'mcat'):
            log_string('both text and visual prompt learner')
            optimizer = torch.optim.Adam(
                [
                    {"params": net.prompt_learner.parameters()},
                    {"params": net.visualpromptlearner.parameters(), "lr": args.lr * res_lr},
                    # {"params": net.visualpromptlearner.firstembedding},
                ],
                lr=args.lr,
                weight_decay=WEIGHT_DECAY,
            )
        else:
            log_string('only text prompt learner')
            optimizer = torch.optim.Adam(
                [
                    {"params": net.prompt_learner.parameters()},
                ],
                lr=args.lr,
                weight_decay=WEIGHT_DECAY,
            )
    else:
        if args.imgp and (args.vptmethod == 'cat' or args.vptmethod == 'macat' or args.vptmethod == 'mcat'):
            log_string('both text and visual prompt learner')
            optimizer = torch.optim.Adam(
                [
                    {"params": net.prompt_learner.parameters()},
                    {"params": net.tasktokenlearner.parameters(), "lr": args.lr * res_lr},
                    {"params": net.visualpromptlearner.parameters()},
                    {"params": net.claposembedding},
                    # {"params": net.visualpromptlearner.firstembedding},
                ],
                lr=args.lr,
                weight_decay=WEIGHT_DECAY,
            )
        elif args.imgp and (args.vptmethod != 'cat' and args.vptmethod != 'macat' and args.vptmethod != 'mcat'):
            log_string('both text and visual prompt learner')
            optimizer = torch.optim.Adam(
                [
                    {"params": net.prompt_learner.parameters()},
                    {"params": net.tasktokenlearner.parameters(), "lr": args.lr * res_lr},
                    {"params": net.visualpromptlearner.parameters()},
                    # {"params": net.visualpromptlearner.firstembedding},
                ],
                lr=args.lr,
                weight_decay=WEIGHT_DECAY,
            )
        else:
            log_string('only text prompt learner')
            optimizer = torch.optim.Adam(
                [
                    {"params": net.prompt_learner.parameters()},
                    {"params": net.tasktokenlearner.parameters()},
                ],
                lr=args.lr,
                weight_decay=WEIGHT_DECAY,
            )
elif OPTIMIZER == "adamw":
    # pdb.set_trace()
    if not args.taskp:
        if args.imgp and (args.vptmethod == 'cat' or args.vptmethod == 'macat' or args.vptmethod == 'mcat'):
            log_string('both text and visual prompt learner')
            optimizer = torch.optim.AdamW(
                [
                    {"params": net.prompt_learner.parameters()},
                    {"params": net.visualpromptlearner.parameters(), "lr": args.lr * res_lr},
                    {"params": net.claposembedding, "lr": args.lr * res_lr},
                    # {"params": net.visualpromptlearner.firstembedding},
                ],
                lr=args.lr,
                weight_decay=WEIGHT_DECAY,
            )
        elif args.imgp and (args.vptmethod != 'cat' and args.vptmethod != 'macat' and args.vptmethod != 'mcat'):
            log_string('both text and visual prompt learner')
            optimizer = torch.optim.AdamW(
                [
                    {"params": net.prompt_learner.parameters()},
                    {"params": net.visualpromptlearner.parameters(), "lr": args.lr * res_lr},
                    # {"params": net.visualpromptlearner.firstembedding},
                ],
                lr=args.lr,
                weight_decay=WEIGHT_DECAY,
            )
        else:
            log_string('only text prompt learner')
            optimizer = torch.optim.AdamW(
                [
                    {"params": net.prompt_learner.parameters()},
                ],
                lr=args.lr,
                weight_decay=WEIGHT_DECAY,
            )
    else:
        if args.imgp and (args.vptmethod == 'cat' or args.vptmethod == 'macat' or args.vptmethod == 'mcat'):
            log_string('both text and visual prompt learner')
            optimizer = torch.optim.AdamW(
                [
                    {"params": net.prompt_learner.parameters()},
                    {"params": net.tasktokenlearner.parameters(), "lr": args.lr * res_lr},
                    {"params": net.visualpromptlearner.parameters()},
                    {"params": net.claposembedding},
                    # {"params": net.visualpromptlearner.firstembedding},
                ],
                lr=args.lr,
                weight_decay=WEIGHT_DECAY,
            )
        elif args.imgp and (args.vptmethod != 'cat' and args.vptmethod != 'macat' and args.vptmethod != 'mcat'):
            log_string('both text and visual prompt learner')
            optimizer = torch.optim.AdamW(
                [
                    {"params": net.prompt_learner.parameters()},
                    {"params": net.tasktokenlearner.parameters(), "lr": args.lr * res_lr},
                    {"params": net.visualpromptlearner.parameters()},
                    # {"params": net.visualpromptlearner.firstembedding},
                ],
                lr=args.lr,
                weight_decay=WEIGHT_DECAY,
            )
        else:
            log_string('only text prompt learner')
            optimizer = torch.optim.AdamW(
                [
                    {"params": net.prompt_learner.parameters()},
                    {"params": net.tasktokenlearner.parameters()},
                ],
                lr=args.lr,
                weight_decay=WEIGHT_DECAY,
            )
else:
    raise NotImplementedError

# pdb.set_trace()

bases_list = [b for a, b in net.named_parameters() if a.endswith("bases")]
other_list = [b for a, b in net.named_parameters() if "coef" not in a]

coef_list = [b for a, b in net.named_parameters() if "coef" in a]
print([a for a, b in net.named_parameters() if "coef" in a])
print([b.shape for a, b in net.named_parameters() if "coef" in a])
log_string("Totally %d coefs." % (len(coef_list)))

# global converge_count
converge_count = 0


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def train(epoch):
    log_string("\nEpoch: %d" % epoch)
    net.train()
    train_loss = 0
    closs = 0
    dloss = 0
    ckl = 0
    correct = 0
    total = 0

    if epoch < 3:
        domain_id = epoch
        loss_rate = 1e-8
    else:
        domain_id = np.random.randint(len(domains))
        loss_rate = 1
    print(domain_id)
    all_dataset.reset("train", domain_id, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        all_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        drop_last=False,
        worker_init_fn=worker_init_fn,
    )

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # pdb.set_trace()
        optimizer.zero_grad()

        if args.imgp:
            # claforimg = targets
            claforimg = torch.cat([targets, torch.randint(54, (20,)).cuda()], 0).unique()
        else:
            claforimg = None

        # pdb.set_trace()

        yc, yd, kld = net(
            inputs, targets, dom=domain_id, featpro=featpro, training=True, mc_times=mc_times, claforimg=claforimg, opentra=args.opentra
        )
        # pdb.set_trace()

        if vpro:
            # pdb.set_trace()
            targets = targets.unsqueeze(-1).repeat(1, mc_times).view(-1)

            common_dloss = criterion(yd.view(-1, len(train_classnames)), targets)

        else:
            common_dloss = torch.zeros(1).cuda()

        # yc = yc * ifcommon
        # common_kl = common_kl * ifcommon
        # ys = ys * ifadapt
        # yps = yps * ifadapt
        # adapt_kl = ifadapt * adapt_kl
        # pdb.set_trace()

        # targets_samples = targets.unsqueeze(1).repeat(1, 10).view(-1)
        results = []
        # results.append(yc.mean(1))

        common_loss = criterion(yc.view(-1, len(train_classnames)), targets)

        mean_results = yc  # .mean(1)
        # ys = []

        # pdb.set_trace()
        # loss = ifcommon*common_loss + ifadapt*adapt_loss + ifadapt*p_beta*prior_loss + ifcommon*c_beta * common_kl + ifadapt*a_beta * adapt_kl
        loss = common_loss + common_dloss + c_beta * kld

        train_loss += loss.item()
        closs += common_loss.item()
        dloss += common_dloss.item()
        ckl += c_beta * kld.item()

        if args.sparse != 0:
            para = 0
            for w in coef_list:
                para = para + torch.sum(torch.abs(w))
            l1_loss = para * args.sparse
            loss = loss + l1_loss

        # pdb.set_trace()
        loss.backward()
        optimizer.step()

        _, mean_preditcted = mean_results.max(1)
        # pdb.set_trace()
        correct += mean_preditcted.eq(targets).sum().item()
        total += targets.size(0)

        if iteration_training and batch_idx >= batchs_per_epoch:
            break

    log_string(
        "Loss: %.3f  | c_loss: %3f | d_loss: %3f | c_kl: %3f | Acc: %.3f%% (%d/%d)"
        % (
            train_loss / (batch_idx + 1),
            closs / (batch_idx + 1),
            dloss / (batch_idx + 1),
            ckl / (batch_idx + 1),
            100.0 * correct / total,
            correct,
            total,
        )
    )

    writer.add_scalar("cls_loss", train_loss / (batch_idx + 1), epoch)
    writer.add_scalar("cls_acc", 100.0 * correct / total, epoch)


def validation(epoch):
    global best_valid_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    # all_dataset.reset('val', 0, transform=transform_test)
    # valloader = torch.utils.data.DataLoader(all_dataset, batch_size=test_batch, shuffle=False, num_workers=4)

    with torch.no_grad():
        for i in range(len(domains)):
            all_dataset.reset("val", i, transform=transform_test)
            valloader = torch.utils.data.DataLoader(
                all_dataset, batch_size=test_batch, shuffle=False, num_workers=4
            )
            for batch_idx, (inputs, targets) in enumerate(valloader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()

                if args.imgp:
                    claforimg = targets
                else:
                    claforimg = None

                yc, yp, _ = net(
                    inputs, targets, dom=i, featpro=featpro, training=False, mc_times=mc_times, claforimg=claforimg
                )
                # ys = ys * ifadapt
                # pdb.set_trace()
                # yc = yc.mean(1, True)
                # yc = yc.mean(1)

                y = torch.softmax(yc, -1)

                # results = []
                # results.append(yc.mean(1))
                # for i in range(len(domains)-1):
                #     results.append(ys[:,:,i].mean(1))
                #     adapt_loss += criterion(ys[:,:,i].view(-1, args.num_classes), targets_samples)

                cls_loss = criterion(y, targets)
                loss = cls_loss

                test_loss += loss.item()
                _, predicted = y.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        log_string(
            "VAL Loss: %.3f | Acc: %.3f%% (%d/%d)"
            % (test_loss / (batch_idx + 1), 100.0 * correct / total, correct, total)
        )

        writer.add_scalar("val_loss", test_loss / (batch_idx + 1), epoch)
        writer.add_scalar("val_acc", 100.0 * correct / total, epoch)
    # Save checkpoint.
    acc = 100.0 * correct / total
    if acc > best_valid_acc:
        print("Saving..")
        log_string("The best validation Acc")
        state = {
            "net": net.state_dict(),
            "acc": acc,
            "epoch": epoch,
        }
        if not os.path.isdir("checkpoint"):
            os.mkdir("checkpoint")
        # torch.save(state, os.path.join(LOG_DIR, "ckpt.t7"))
        best_valid_acc = acc
        return 0
    else:
        return 1


def test(epoch):
    global best_acc
    global best_accp
    net.eval()
    test_loss = 0
    test_lossp = 0
    correct = 0
    correctp = 0
    total = 0
    dom_total = 0
    dom_correct = 0
    dom_acc = []

    all_dataset.reset("test", 0, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        all_dataset, batch_size=test_batch, shuffle=False, num_workers=4
    )

    # center_feature = np.zeros(7, 512)
    # pdb.set_trace()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(testloader)):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            if args.imgp:
                claforimg = targets
            else:
                claforimg = None

            # pdb.set_trace()

            yc, yp, _ = net(
                inputs, targets, dom=3, featpro=featpro, training=False, mc_times=mc_times, claforimg=claforimg
            )

            # yc = yc.mean(1)

            y = torch.softmax(yc, -1)
            yp = torch.softmax(yp, -1)

            cls_loss = criterion(y, targets)
            cls_lossp = criterion(yp, targets)
            loss = cls_loss

            test_loss += loss.item()
            test_lossp += cls_lossp.item()
            _, predicted = y.max(1)
            _, predictedp = yp.max(1)
            total += targets.size(0)
            dom_total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            correctp += predictedp.eq(targets).sum().item()
            dom_correct += predicted.eq(targets).sum().item()

            if (args.dataset == "FM" or args.dataset == "mnist") and (
                batch_idx + 1
            ) % 100 == 0:
                dom_acc.append(100.0 * dom_correct / dom_total)
                dom_total = 0
                dom_correct = 0

        log_string(
            "TEST Loss: %.3f | Acc: %.3f%% (%d/%d)"
            % (test_loss / (batch_idx + 1), 100.0 * correct / total, correct, total)
        )

        log_string(
            "TESTp Loss: %.3f | Acc: %.3f%% (%d/%d)"
            % (test_lossp / (batch_idx + 1), 100.0 * correctp / total, correctp, total)
        )

        if args.dataset == "FM" or args.dataset == "mnist":
            for i in range(len(test_domain)):
                log_string("domain %s Acc: %.3f" % (test_domain[i], dom_acc[i]))

        writer.add_scalar("test_loss", test_loss / (batch_idx + 1), epoch)
        writer.add_scalar("test_acc", 100.0 * correct / total, epoch)

    # Save checkpoint.
    acc = 100.0 * correct / total
    accp = 100.0 * correctp / total
    if accp > best_accp:
        log_string("The best testp Acc")

        best_accp = accp


    if acc > best_acc:
        print("Saving..")
        log_string("The best test Acc")
        state = {
            "net": net.state_dict(),
            "acc": acc,
            "epoch": epoch,
        }
        if not os.path.isdir("checkpoint"):
            os.mkdir("checkpoint")
        torch.save(state, os.path.join(LOG_DIR, "ckpt.t7"))
        best_acc = acc
        return 0
    else:
        return 1


# decay_ite = [0.6*max_ite]
decay_ite = []
if args.autodecay:
    for epoch in range(300):
        train(epoch)
        f = test(epoch)
        if f == 0:
            converge_count = 0
        else:
            converge_count += 1

        if converge_count == 20:
            optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"] * 0.2
            log_string(
                "In epoch %d the LR is decay to %f"
                % (epoch, optimizer.param_groups[0]["lr"])
            )
            converge_count = 0

        if optimizer.param_groups[0]["lr"] < 2e-6:
            exit()

else:
    if not iteration_training:
        for epoch in range(start_epoch, start_epoch + decay_inter[-1] + 50):
            if epoch in decay_inter:
                optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"] * 0.1
                log_string(
                    "In epoch %d the LR is decay to %f"
                    % (epoch, optimizer.param_groups[0]["lr"])
                )
            train(epoch)
            if epoch % 5 == 0:
                if args.dataset != "office":
                    _ = validation(epoch)
                # _ = validation(epoch)
                _ = test(epoch)
    else:
        if trainpro:
            for epoch in range(max_ite):
                if epoch in decay_ite:
                    for i in range(len(optimizer.param_groups)):
                        optimizer.param_groups[i]["lr"] = (
                            optimizer.param_groups[i]["lr"] * 0.1
                        )
                    log_string(
                        "In iteration %d the LR is decay to %f"
                        % (epoch, optimizer.param_groups[0]["lr"])
                    )
                train(epoch)
                if epoch % test_ite == 0 and epoch!=0:
                    # _ = validation(epoch)
                    if args.dataset != "office" and args.dataset != "domainnet":
                        _ = validation(epoch)
                    _ = test(epoch)

        else:
            _ = test(0)
