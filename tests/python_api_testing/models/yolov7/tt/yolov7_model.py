import torch
from torch import nn
import tt_lib
import warnings
from copy import deepcopy
from loguru import logger
from pathlib import Path
import contextlib
import math

from python_api_testing.models.yolov7.tt.yolov7_conv import TtConv
from python_api_testing.models.yolov7.tt.yolov7_upsample import TtUpsample
from python_api_testing.models.yolov7.tt.yolov7_concat import TtConcat
from python_api_testing.models.yolov7.tt.yolov7_detect import TtDetect
from python_api_testing.models.yolov7.tt.yolov7_repconv import TtRepConv
from python_api_testing.models.yolov7.tt.yolov7_sppcspc import TtSPPCSPC
from python_api_testing.models.yolov7.tt.yolov7_mp import TtMP, TtMaxPool2D
from tt_lib.fallback_ops import fallback_ops

from python_api_testing.models.yolov7.reference.utils.general import make_divisible
from python_api_testing.models.yolov7.reference.utils.autoanchor import (
    check_anchor_order,
)
from python_api_testing.models.yolov7.reference.utils.torch_utils import (
    fuse_conv_and_bn,
    model_info,
    profile,
    scale_img,
)
from python_api_testing.models.yolov7.reference.models.load_torch_model import (
    get_yolov7_fused_cpu_model,
)
from utility_functions_new import torch2tt_tensor, tt2torch_tensor


def parse_model(state_dict, base_address, yaml_dict, ch, device):
    # model_dict, input_channels(3)
    d = yaml_dict

    logger.info(
        "\n%3s%18s%3s%10s  %-40s%-30s"
        % ("", "from", "n", "params", "module", "arguments")
    )
    anchors, nc, gd, gw = (
        d["anchors"],
        d["nc"],
        d["depth_multiple"],
        d["width_multiple"],
    )
    na = (
        (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
    )  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out

    for i, (f, n, m, args) in enumerate(
        d["backbone"] + d["head"]
    ):  # from, number, module, args
        # m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [
            "Conv",
            "RepConv",
            "SPPCSPC",
        ]:
            c1, c2 = ch[f], args[0]

            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]

            if m == "SPPCSPC":
                args.insert(2, n)  # number of repeats
                n = 1

            m = eval(f"Tt{m}")

        elif m == "Concat":
            c2 = sum([ch[x] for x in f])
            m = TtConcat

        elif m == "Detect":
            args.append([ch[x] for x in f])
            m = TtDetect

            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)

        elif m == "nn.Upsample":
            # maybe tthis block needed as well
            c2 = ch[f]
            m = TtUpsample

        elif m == "MaxPool2d":
            # maybe tthis block needed as well
            c2 = ch[f]
            m = TtMaxPool2D
        elif m == "MP":
            # maybe tthis block needed as well
            c2 = ch[f]
            m = TtMP

        else:
            c2 = ch[f]

        args.insert(0, f"{base_address}.{i}")
        args.insert(0, state_dict)
        args.insert(0, device)

        if n > 1:
            list_modules = []
            for iter_layer in range(n):
                args[2] = f"{base_address}.{i}.{iter_layer}"
                list_modules.append((m(*args)))

            m_ = nn.Sequential(*list_modules)
        else:
            m_ = m(*args)

        t = str(m)[8:-2].replace("__main__.", "")  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = (
            i,
            f,
            t,
            np,
        )  # attach index, 'from' index, type, number params

        save.extend(
            x % i for x in ([f] if isinstance(f, int) else f) if x != -1
        )  # append to savelist
        layers.append(m_)

        if i == 0:
            ch = []

        ch.append(c2)

    return nn.Sequential(*layers), sorted(save)


class TtModel(nn.Module):
    def __init__(
        self,
        device,
        state_dict,
        base_address,
        cfg="yolov7.yaml",
        ch=3,
        nc=None,
        anchors=None,
    ):
        # model, input channels, number of classes
        super().__init__()
        self.device = device
        self.traced = False
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub

            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.SafeLoader)  # model dict

        # Define model
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels

        if nc and nc != self.yaml["nc"]:
            logger.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override yaml value
        if anchors:
            logger.info(f"Overriding model.yaml anchors with anchors={anchors}")
            self.yaml["anchors"] = round(anchors)  # override yaml value

        self.model, self.save = parse_model(
            state_dict, base_address, deepcopy(self.yaml), ch=[ch], device=self.device
        )  # model, savelist

        self.names = [str(i) for i in range(self.yaml["nc"])]  # default names

        # # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, TtDetect):
            s = 256  # 2x min stride
            m.stride = torch.tensor(
                [
                    s / x.shape[-2]
                    for x in self.forward(
                        torch2tt_tensor(torch.zeros(1, ch, s, s), self.device)
                    )
                ]
            )  # forward
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride

        self.info()

    def forward(self, x, augment=False, profile=False):
        if augment:
            raise NotImplementedError
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
                yi = self.forward_once(xi)[0]  # forward
                # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            return self.forward_once(x, profile)  # single-scale inference, train

    def forward_once(self, x, profile=False):
        y, dt = [], []  # outputs
        for iteration, m in enumerate(self.model):
            if m.f != -1:  # if not from previous layer
                x = (
                    y[m.f]
                    if isinstance(m.f, int)
                    else [x if j == -1 else y[j] for j in m.f]
                )  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        return x

    def _initialize_biases(self, cf=None):
        raise NotImplementedError
        # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(
                8 / (640 / s) ** 2
            )  # obj (8 objects per 640 image)
            b.data[:, 5:] += (
                math.log(0.6 / (m.nc - 0.99))
                if cf is None
                else torch.log(cf / cf.sum())
            )  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _initialize_aux_biases(self, cf=None):
        raise NotImplementedError
        # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, mi2, s in zip(m.m, m.m2, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(
                8 / (640 / s) ** 2
            )  # obj (8 objects per 640 image)
            b.data[:, 5:] += (
                math.log(0.6 / (m.nc - 0.99))
                if cf is None
                else torch.log(cf / cf.sum())
            )  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            b2 = mi2.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b2.data[:, 4] += math.log(
                8 / (640 / s) ** 2
            )  # obj (8 objects per 640 image)
            b2.data[:, 5:] += (
                math.log(0.6 / (m.nc - 0.99))
                if cf is None
                else torch.log(cf / cf.sum())
            )  # cls
            mi2.bias = torch.nn.Parameter(b2.view(-1), requires_grad=True)

    def _initialize_biases_bin(self, cf=None):
        raise NotImplementedError
        # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Bin() module
        bc = m.bin_count
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            old = b[:, (0, 1, 2, bc + 3)].data
            obj_idx = 2 * bc + 4
            b[:, :obj_idx].data += math.log(0.6 / (bc + 1 - 0.99))
            b[:, obj_idx].data += math.log(
                8 / (640 / s) ** 2
            )  # obj (8 objects per 640 image)
            b[:, (obj_idx + 1) :].data += (
                math.log(0.6 / (m.nc - 0.99))
                if cf is None
                else torch.log(cf / cf.sum())
            )  # cls
            b[:, (0, 1, 2, bc + 3)].data = old
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _initialize_biases_kpt(self, cf=None):
        raise NotImplementedError
        # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(
                8 / (640 / s) ** 2
            )  # obj (8 objects per 640 image)
            b.data[:, 5:] += (
                math.log(0.6 / (m.nc - 0.99))
                if cf is None
                else torch.log(cf / cf.sum())
            )  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        raise NotImplementedError
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            logger.info(
                ("%6g Conv2d.bias:" + "%10.3g" * 6)
                % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean())
            )

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        raise NotImplementedError
        logger.info("Fusing layers... ")
        for m in self.model.modules():
            if "RepConv" in str(type(m)):
                # logger.info(f" fuse_repvgg_block")
                m.fuse_repvgg_block()
            elif isinstance(m, RepConv_OREPA):
                # logger.info(f" switch_to_deploy")
                m.switch_to_deploy()
            # elif type(m) is Conv and hasattr(m, 'bn'):
            elif hasattr(m, "bn") and hasattr(m, "conv"):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, "bn")  # remove batchnorm
                m.forward = m.fuseforward  # update forward
            elif isinstance(m, (IDetect, IAuxDetect)):
                m.fuse()
                m.forward = m.fuseforward
        self.info()
        return self

    def nms(self, mode=True):  # add or remove NMS module
        raise NotImplementedError
        present = type(self.model[-1]) is NMS  # last layer is NMS
        if mode and not present:
            logger.info("Adding NMS... ")
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name="%s" % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            logger.info("Removing NMS... ")
            self.model = self.model[:-1]  # remove
        return self

    def autoshape(self):  # add autoShape module
        raise NotImplementedError
        logger.info("Adding autoShape... ")
        m = autoShape(self)  # wrap model
        copy_attr(
            m, self, include=("yaml", "nc", "hyp", "names", "stride"), exclude=()
        )  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # logger.info model information
        model_info(self, verbose, img_size)


def _yolov7_fused_model(cfg_path, state_dict, base_address, device) -> TtModel:
    tt_model = TtModel(
        device=device,
        state_dict=state_dict,
        base_address=base_address,
        cfg=cfg_path,
    )
    return tt_model


def yolov7_fused_model(device, model_location_generator) -> TtModel:
    model_path = model_location_generator("tt_dnn-models/Yolo/models/")
    cfg_data_path = model_location_generator("tt_dnn-models/Yolo/data/")
    cfg_path = str(cfg_data_path / "yolov7.yaml")
    weights = str(model_path / "yolov7.pt")
    reference_model = get_yolov7_fused_cpu_model(
        model_location_generator
    )  # load FP32 model

    tt_model = _yolov7_fused_model(
        cfg_path=cfg_path,
        state_dict=reference_model.state_dict(),
        base_address="model",
        device=device,
    )

    return tt_model
