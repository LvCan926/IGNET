import torch
from copy import deepcopy
from yacs.config import CfgNode
from typing import Dict
from pathlib import Path


def Build_TP_model(cfg: CfgNode, args):
    if "fastpredNF" in cfg.MODEL.TYPE:
        from models.TP.fastpredNF import fastpredNF_TP
        
        from models.TP.gen_flow import TrajectoryFlowGenerator

        # model = fastpredNF_TP(cfg, args).cuda()

        model = TrajectoryFlowGenerator(cfg, args).cuda()
        print("TrajectoryFlowGenerator model is built")

    else:
        raise (ValueError, f"unknown model type: {cfg.MODEL.TYPE}")

    return model


class GT_Dist(torch.nn.Module):
    def __init__(self, cfg: CfgNode) -> None:
        super(GT_Dist, self).__init__()
        self.pred_len = cfg.DATA.PREDICT_LENGTH

        self.task = cfg.DATA.TASK

        import dill

        env_path = (
            Path(cfg.DATA.PATH)
            / cfg.DATA.TASK
            / "processed_data"
            / f"{cfg.DATA.DATASET_NAME}_train.pkl"
        )
        with open(env_path, "rb") as f:
            train_env = dill.load(f, encoding="latin1")

        from models.TP.components.gmm2d import GMM2D

        gt_dist = train_env.gt_dist
        assert gt_dist is not None, "environment does not have GT distributions"
        gt_dist = gt_dist.transpose(1, 0, 2)
        gt_dist = gt_dist[-self.pred_len :]
        L, N, D = gt_dist.shape
        log_pis = torch.log(torch.ones(1, 1, L, N) / N)
        mus = torch.Tensor(gt_dist[None, None, ..., :2])
        log_sigmas = torch.log(torch.Tensor(gt_dist[None, None, ..., 2:]))
        corrs = torch.zeros(1, 1, L, N)

        self.kernels = GMM2D(
            log_pis.cuda(), mus.cuda(), log_sigmas.cuda(), corrs.cuda()
        )

        self.obss = []
        self.gts = []

    def predict(self, data_dict, return_prob=False):
        data_dict[("pred", 0)] = deepcopy(data_dict["gt"])
        self.obss.append(data_dict["obs"].cpu().numpy())
        self.gts.append(data_dict["gt"].cpu().numpy())
        if return_prob:
            data_dict[("prob", 0)] = self.kernels
        return data_dict

    def predict_from_new_obs(self, data_dict: Dict, time_step: int) -> Dict:
        return data_dict

    def update(self, data_dict):
        pass

    def load(self, path: Path = None) -> bool:
        pass

    def save(self, epoch: int = 0, path: Path = None) -> int:
        pass
