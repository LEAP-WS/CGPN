# CGPN
This is the repository for the NeurIPS-21 paper [Contrastive Graph Poisson Networks: Semi-Supervised Learning with Extremely Limited Labels].

Abstract: Graph Neural Networks (GNNs) have achieved remarkable performance in the task of semi-supervised node classification. However, most existing GNN models require sufficient labeled data for effective network training. Their performance can be seriously degraded when labels are extremely limited. To address this issue, we propose a new framework termed Contrastive Graph Poisson Networks (CGPN) for node classification under extremely limited labeled data. Specifically, our CGPN derives from variational inference; integrates a newly designed Graph Poisson Network (GPN) to effectively propagate the limited labels to the entire graph and a normal GNN, such as Graph Attention Network, that flexibly guides the propagation of GPN; applies a contrastive objective to further exploit the supervision information from the learning process of GPN and GNN models. Essentially, our CGPN can enhance the learning performance of GNNs under extremely limited labels by contrastively propagating the limited labels to the entire graph. We conducted extensive experiments on different types of datasets to demonstrate the superiority of CGPN. 

## Requirements

- PyTorch (1.4.0)

## Usage

You can conduct node classification experiments on benchmark datasets (e.g., Cora) by running the 'main.py' file.

## Cite
Please cite our paper if you use this code in your own work:

```
@inproceedings{Wan2021Contrastive,
  title={Contrastive Graph Poisson Networks: Semi-Supervised Learning with Extremely Limited Labels},
  author={Wan, Sheng and Zhan, Yibing and Liu, Liu and Yu, Baosheng and Pan, Shirui and Gong, Chen},
  booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
  year={2021}
}
```
