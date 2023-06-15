import sys
sys.path.append('/data/projects/fate/persistence/torch-rechub')

import argparse
import os
import torch
from torch import nn
from torch_rechub.basic.layers import FM,LR,EmbeddingLayer
from torch_rechub.basic.features import DenseFeature, SparseFeature
from federatedml.nn.dataset.criteo_dataset import CriteoDataset
import numpy as np
import pandas as pd
from pipeline.interface import Data, Model
from pipeline.component import Reader, Evaluation, DataTransform
from pipeline.backend.pipeline import PipeLine
from pipeline.component import HomoNN
from pipeline import fate_torch_hook

from pipeline.component.homo_nn import DatasetParam, TrainerParam 
from pipeline.utils.tools import load_job_config


class FMModel(torch.nn.Module):
  '''
  A standard FM models
  '''
  def __init__(self, dense_feas_dict, sparse_feas_dict):
      super(FMModel, self).__init__()
      dense_features = []
      def recover_dense_feat(dict_):
        return DenseFeature(name=dict_['name'])
      
      for i in dense_feas_dict:
        dense_features.append(recover_dense_feat(i))
      self.dense_features = dense_features
      
      sparse_features = []
      def recover_sparse_feat(dict_):
          return SparseFeature(dict_['name'], dict_['vocab_size'],dict_['embed_dim'])
      
      for i in sparse_feas_dict:
        sparse_features.append(recover_sparse_feat(i))
      self.sparse_features = sparse_features
      
      self.fm_features = self.sparse_features
      self.fm_dims = sum([fea.embed_dim for fea in self.fm_features])
      self.linear = LR(self.fm_dims)  # 1-odrder interaction
      self.fm = FM(reduce_sum=True)  # 2-odrder interaction
      self.embedding = EmbeddingLayer(self.fm_features)

  def forward(self, x):
      input_fm = self.embedding(x, self.fm_features, squeeze_dim=False)  #[batch_size, num_fields, embed_dim]input_fm: torch.Size([100, 39, 16])
      # print('input_fm:',input_fm.shape) # input_fm: torch.Size([100, 39, 16]) (batch_size, num_features, embed_dim)
      y_linear = self.linear(input_fm.flatten(start_dim=1))
      y_fm = self.fm(input_fm)
      # print('y_fm.shape:',y_fm.shape) # y_fm.shape: torch.Size([100, 1])

      y = y_linear + y_fm
      return torch.sigmoid(y.squeeze(1))


def main(config="/data/projects/fate/examples/config.yaml", namespace=""):
    # obtain config
    if isinstance(config, str):
        config = load_job_config(config)
        
    fate_torch_hook(torch)

    fate_project_path = os.path.abspath('../../../../')
    data_path = 'examples/data/criteo.csv'
    host = 10000
    guest = 9999
    arbiter = 10000
    pipeline = PipeLine().set_initiator(role='guest', party_id=guest).set_roles(guest=guest, host=host, arbiter=arbiter)

    data = {"name": "criteo", "namespace": "experiment"}
    pipeline.bind_table(name=data['name'],
                        namespace=data['namespace'], path=fate_project_path + '/' + data_path)

    # reader
    reader_0 = Reader(name="reader_0")
    reader_0.get_party_instance(
        role='guest', party_id=guest).component_param(table=data)
    reader_0.get_party_instance(
        role='host', party_id=host).component_param(table=data)

    ds = CriteoDataset()
    dense_feas, dense_feas_dict, sparse_feas, sparse_feas_dict, ffm_linear_feas, ffm_linear_feas_dict, \
            ffm_cross_feas, ffm_cross_feas_dict = ds.load(fate_project_path + '/' + data_path)

    model = torch.nn.Sequential(
        torch.nn.CustModel(module_name='fm_model', class_name='FMModel', dense_feas_dict=dense_feas_dict, sparse_feas_dict=sparse_feas_dict)
    )

    nn_component = HomoNN(name='nn_0',
                        model=model, 
                        loss=torch.nn.BCELoss(),
                        optimizer=torch.optim.Adam(
                            model.parameters(), lr=0.001, weight_decay=0.001),
                        dataset=DatasetParam(dataset_name='criteo_dataset'),
                        trainer=TrainerParam(trainer_name='fedavg_trainer', epochs=1, batch_size=256, validation_freqs=1,
                                            data_loader_worker=6, shuffle=True),
                        torch_seed=100 
                        )

    pipeline.add_component(reader_0)
    pipeline.add_component(nn_component, data=Data(train_data=reader_0.output.data))
    pipeline.add_component(Evaluation(name='eval_0', eval_type='binary'), data=Data(data=nn_component.output.data))
    pipeline.compile()
    pipeline.fit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("CTR DEMO")
    parser.add_argument("-config", type=str,
                        help="config file")
    args = parser.parse_args()
    if args.config is not None:
        main(args.config)
    else:
        main()