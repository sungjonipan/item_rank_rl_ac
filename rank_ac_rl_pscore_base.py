from collections import OrderedDict
from functools import reduce

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.framework.python.ops import variables as contrib_variables
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope

from model_ops import checkpoint_utils
from model_ops import metrics
from model_ops import ops as base_ops
from model_ops import utils
from model_ops.tflog import tflogger as logging
from model_zoo.base_model import BaseModel
from model_zoo.yuexia_ac_rl_transformer_base import YuexiaACRLTransformerBase


class YuexiaACRLTransformerCritic(YuexiaACRLTransformerBase):
    def __init__(self,
                 model_config,
                 training_config,
                 mc,
                 fg,
                 context,
                 name="CTR"):
        super(YuexiaACRLTransformerCritic, self).__init__(
            model_config,
            training_config,
            mc,
            fg,
            context,
            name)
        logging.info('[YuexiaACRLTransformerCritic | init] init start.')
        # Dnn net
        self.label_detail = None
        self.label_ord = None

        self.column_blocks = ['account_columns'] + self.main_column_blocks + self.bias_column_blocks
        self.seq_column_blocks = [] + self.ta_seq_column_blocks
        self.column_blocks += [seq_name + '_length' for seq_name in self.seq_column_blocks]

        logging.info("model: {}, column_blocks: {}".format(self.name, self.column_blocks))
        logging.info("model: {}, seq_column_blocks: {}".format(self.name, self.seq_column_blocks))
        logging.info("model: {}, main_column_blocks: {}".format(self.name, self.main_column_blocks))
        logging.info("model: {}, bias_column_blocks: {}".format(self.name, self.bias_column_blocks))

        self.mtl_task = self.training_config.mtl_task
        self.mtl_label = {}
        self.mtl_main_net = {}
        self.mtl_bias_net = {}
        self.mtl_bias = {}
        self.mtl_main_logits = {}
        self.mtl_bias_logits = {}
        self.mtl_bias = {}
        self.mtl_logits = {}
        self.mtl_predictions = {}
        self.mtl_ce_loss = {}
        self.mtl_ranking_loss = {}
        self.mtl_loss = {}
        self.mtl_auc = {}
        self.mtl_total_auc = {}
        self.seq_real_len = {}
        self.mtl_loss_weight = self.training_config.mtl_loss_weight
        self.esmm_dict = self.training_config.esmm_dict
        self.clkTaskName = self.training_config.clkTaskName

        self.seq_block_layer_dict = {}

    def build_inputs(self, features, feature_columns, labels):
        logging.info('[YuexiaACRLTransformerCritic | build_inputs] build_inputs start.')
        super(YuexiaACRLTransformerCritic, self).build_inputs(features, feature_columns, labels)
        self.label = tf.identity(self.label)

        for labelName in self.mtl_task:
            logging.info("[YX DEBUG | build_inputs] labelName: {}".format(labelName))
            if labelName == 'is_item_click':
                cur_label = tf.identity(tf.cast(tf.logical_or(tf.greater(self.features["is_item1_click"], 0.0),
                                                              tf.greater(self.features["is_item2_click"], 0.0)),
                                                dtype=tf.float32))
            elif labelName == 'is_item_ord_strict':
                cur_label = tf.identity(tf.cast(tf.logical_or(tf.greater(self.features["label_item1_ord_strict"], 0.0),
                                                              tf.greater(self.features["label_item2_ord_strict"], 0.0)),
                                                dtype=tf.float32))
            elif labelName == 'is_item_ord':
                cur_label = tf.identity(tf.cast(tf.logical_or(tf.greater(self.features["label_item1_ord"], 0.0),
                                                              tf.greater(self.features["label_item2_ord"], 0.0)),
                                                dtype=tf.float32))

            elif labelName == 'is_item2feed_ord':
                is_item_click = tf.identity(tf.cast(tf.logical_or(tf.greater(self.features["is_item1_click"], 0.0),
                                                                  tf.greater(self.features["is_item2_click"], 0.0)),
                                                    dtype=tf.float32))
                cur_label = tf.identity(tf.cast(tf.logical_and(tf.greater(is_item_click, 0.0),
                                                               tf.greater(self.features["label_feed_ord_strict"], 0.0)),
                                                dtype=tf.float32))
            else:
                cur_label = tf.identity(self.features[labelName])

            self.mtl_label[labelName] = tf.identity(cur_label)

    def build_model(self):
        logging.info('[YuexiaACRLTransformerCritic | build_model] build_model start.')
        if self.training_config.use_gpu:
            with tf.device('/CPU:0'):
                self.embedding_layer()
            with tf.device('/GPU:0'):
                self.seq_pooling_layer()
                self.seq_target_atten_layer()
                self.latent_condition_transform()
                self.bias_net()
                self.transformer_layer()
                self.logits_layer()
        else:
            self.embedding_layer()
            self.seq_pooling_layer()
            self.seq_target_atten_layer()
            self.latent_condition_transform()
            self.bias_net()
            self.transformer_layer()
            self.logits_layer()

    def loss_op(self):
        logging.info('[YuexiaACRLTransformerCritic | loss_op] loss_op start.')
        with tf.name_scope("{}_Loss_Op".format(self.name)):
            """
            weighted_sigmoid_cross_entropy_with_logits.
            x = logits, z = labels, q = pos_weight
            loss = q * z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
                 = q * z * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
                 = q * z * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
                 = q * z * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
                 = (1 - z) * x + (1 + (q - 1) * z) * log(1 + exp(-x))
                 = (1 - z) * x + [1 + (q - 1) * z] * [log(1 + exp(-abs(x)) + max(-x, 0)]
            """
            self.loss = self.reg_loss
            for taskName in self.mtl_task:
                loss_weight = self.mtl_loss_weight[taskName]
                if taskName in self.esmm_dict:
                    clickTask = self.esmm_dict[taskName]
                    eps = 1e-8
                    ctr_pred = tf.sigmoid(self.mtl_logits[clickTask])
                    cvr_pred = tf.sigmoid(self.mtl_logits[taskName])
                    ctr_pred = tf.clip_by_value(ctr_pred, eps, 0.9999)
                    cvr_pred = tf.clip_by_value(cvr_pred, eps, 0.9999)
                    impr_cvr_pred = tf.stop_gradient(ctr_pred) * cvr_pred
                    impr_cvr_pred_clip = tf.clip_by_value(impr_cvr_pred, eps, 0.9999)
                    impr_cvr_label = self.mtl_label[taskName]
                    self.mtl_ce_loss[taskName] = -1.0 * tf.reduce_mean(
                        impr_cvr_label * tf.log(impr_cvr_pred_clip) + (1 - impr_cvr_label) * tf.log(
                            1 - impr_cvr_pred_clip))
                    if self.config.ranking_loss_weights.get(taskName, 0.0) > 0.0:
                        logging.info("task_name: {}, use ranking_loss.".format(taskName))
                        task_rank_loss = self.cal_rank_loss(tf.expand_dims(self.mtl_logits[taskName], axis=-1),
                                                            tf.expand_dims(impr_cvr_label, axis=-1))
                        self.mtl_ranking_loss[taskName] = task_rank_loss
                else:
                    self.mtl_ce_loss[taskName] = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=self.mtl_logits[taskName], labels=self.mtl_label[taskName]))
                    if self.config.ranking_loss_weights.get(taskName, 0.0) > 0.0:
                        logging.info("task_name: {}, use ranking_loss.".format(taskName))
                        task_rank_loss = self.cal_rank_loss(tf.expand_dims(self.mtl_logits[taskName], axis=-1),
                                                            tf.expand_dims(self.mtl_label[taskName], axis=-1))
                        self.mtl_ranking_loss[taskName] = task_rank_loss
                self.loss += loss_weight * self.mtl_ce_loss[taskName]
                if self.config.ranking_loss_weights.get(taskName, 0.0) > 0.0:
                    self.loss += self.config.ranking_loss_weights.get(taskName) * self.mtl_ranking_loss[taskName]

            self.ce_loss = self.mtl_ce_loss[self.clkTaskName]

    def setup_global_step(self):
        logging.info('[YuexiaACRLTransformerCritic | setup_global_step] setup_global_step start.')
        global_step = tf.Variable(
            initial_value=0,
            name="{}_global_step".format(self.name),
            trainable=False,
            dtype=tf.int64,
            collections=[tf.GraphKeys.GLOBAL_VARIABLES])

        self.global_step = global_step
        self.global_step_reset = tf.assign(self.global_step, 0)
        self.global_step_add = tf.assign_add(self.global_step, 1, use_locking=True)
        tf.summary.scalar('global_step/' + self.global_step.name, self.global_step)

    def predictions_op(self):
        logging.info('[YuexiaACRLTransformerCritic | predictions_op] predictions_op start.')
        with tf.name_scope("{}_Predictions_Op".format(self.name)):
            self.predictions = tf.sigmoid(self.logits)
            for taskName in self.mtl_task:
                self.mtl_predictions[taskName] = tf.sigmoid(self.mtl_logits[taskName])

    def get_prediction_map(self):
        return self.mtl_predictions

    def get_label_map(self):
        return self.mtl_label

    def mark_output(self):
        with tf.name_scope("{}_Mark_Output".format(self.name)):
            logistic = tf.identity(self.predictions, name="rank_predict")
            for taskName in self.mtl_task:
                logistic = tf.identity(self.mtl_predictions[taskName], name="{}_rank_predict".format(taskName))
            rank_item1_index = tf.identity(self.features['rank_item1_index'], name="item1_index_rank_predict")
            rank_item2_index = tf.identity(self.features['rank_item2_index'], name="item2_index_rank_predict")

    def metrics_op(self):
        with tf.name_scope("{}_Metrics".format(self.name)):
            with tf.device(self.worker_device):
                self.auc, self.total_auc = metrics.auc(
                    labels=self.label,
                    predictions=self.predictions,
                    num_thresholds=2000)

                for taskName in self.mtl_task:
                    self.mtl_auc[taskName], self.mtl_total_auc[taskName] = metrics.auc(
                        labels=self.mtl_label[taskName],
                        predictions=self.mtl_predictions[taskName],
                        num_thresholds=2000)

            # scalar
            self.metrics['scalar/ce_loss'] = self.ce_loss
            self.metrics['scalar/auc'] = self.auc
            self.metrics['scalar/total_auc'] = self.total_auc
            self.metrics['scalar/label_mean'] = tf.reduce_mean(self.label)
            self.metrics['scalar/logits_mean'] = tf.reduce_mean(self.logits)
            self.metrics['scalar/predictions_mean'] = tf.reduce_mean(self.predictions)

            for taskName in self.mtl_task:
                self.metrics['scalar/{}_ce_loss'.format(taskName)] = self.mtl_ce_loss[taskName]
                self.metrics['scalar/{}_auc'.format(taskName)] = self.mtl_auc[taskName]
                self.metrics['scalar/{}_total_auc'.format(taskName)] = self.mtl_total_auc[taskName]
                self.metrics['scalar/{}_label_mean'.format(taskName)] = tf.reduce_mean(self.mtl_label[taskName])
                self.metrics['scalar/{}_logits_mean'.format(taskName)] = tf.reduce_mean(self.mtl_logits[taskName])
                self.metrics['scalar/{}_predictions_mean'.format(taskName)] = tf.reduce_mean(
                    self.mtl_predictions[taskName])
                if taskName in self.mtl_ranking_loss.keys():
                    self.metrics['scalar/{}_ranking_loss'.format(taskName)] = self.mtl_ranking_loss[taskName]

            self.metrics['scalar/reg_loss'] = self.reg_loss
            self.metrics['scalar/loss'] = self.loss

            # set total auc in model ops for model eval
            self.context.add_validate_op(self.total_auc)
            self.context.add_validate_op(tf.reduce_mean(self.predictions))
            # set total auc in model ops for odl ckpt model eval
            self.context.get_model_ops().set_global_auc(self.total_auc)
            self.context.get_model_ops().set_current_auc(self.auc)

    def build(self, features, feature_columns, labels):
        logging.info('[YuexiaACRLTransformerCritic | build] build start.')
        super(YuexiaACRLTransformerCritic, self).build(features, feature_columns, labels)
        if self.training_config.model_checkpoint_dir is not None and self.training_config.model_checkpoint_dir.strip() != "":
            logging.info("loading the checkpoint: {}, {}".format(self.config.model_checkpoint_dir,
                                                                 self.config.restore_var_scope))
            checkpoint_utils.restore_from_checkpoint(ckpt_dir_or_file=self.config.model_checkpoint_dir,
                                                     restore_var_scope=self.config.restore_var_scope)

    def _get_predictions_schema(self):
        schemas = ["id", "logits", "predictions"]
        for taskName in self.mtl_task:
            schemas.append(taskName + "_label")
            schemas.append(taskName + "_pred")
        types = ["STRING"] + ["DOUBLE"] * (len(schemas) - 1)
        return schemas, types

    def _get_predictions_op(self):
        predictions = [self.id, self.logits, self.predictions]
        for taskName in self.mtl_task:
            predictions.append(self.mtl_label[taskName])
            predictions.append(self.mtl_predictions[taskName])
        return predictions

    def cal_rank_loss(self, logits, labels):
        logging.info('[YuexiaACRLTransformerCritic | cal_rank_loss] cal_rank_loss start.')
        pairwise_logits = logits - tf.transpose(logits)
        logging.info("[rank_loss] pairwise logits: {}".format(pairwise_logits))
        pairwise_mask = tf.greater(labels - tf.transpose(labels), 0)
        logging.info("[rank_loss] mask: {}".format(pairwise_mask))
        pairwise_logits = tf.boolean_mask(pairwise_logits, pairwise_mask)
        logging.info("[rank_loss]: after masking: {}".format(pairwise_logits))
        pairwise_psudo_labels = tf.ones_like(pairwise_logits)
        rank_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=pairwise_logits,
            labels=pairwise_psudo_labels
        ))
        # set rank loss to zero if a batch has no positive sample.
        rank_loss = tf.where(tf.is_nan(rank_loss), tf.zeros_like(rank_loss), rank_loss)
        return rank_loss

    def summary_op(self):
        super(YuexiaACRLTransformerCritic, self).summary_op()
        with tf.name_scope('{}_Sequence_Embedding_Summary'.format(self.name)):
            for block_name, layer in self.seq_block_layer_dict.items():
                if not self.mc.seq_has_block(block_name):
                    continue
                seq_len = self.fg.get_seq_len_by_sequence_name(block_name)
                seq_real_length_name = '{}_length'.format(block_name)
                seq_real_length = self.block_layer_dict[seq_real_length_name]
                seq_real_length = tf.minimum(seq_real_length, tf.ones_like(seq_real_length) * seq_len)
                base_ops.add_seq_embed_layer_norm(tf.reshape(layer, [-1, tf.shape(layer)[2]]),
                                                  self.feature_columns[block_name],
                                                  seq_real_length,
                                                  self.config.fix_sorted_columns)
                logging.info('seq embedding block_name: {} and columns: {}'.format(block_name,
                                                                                   self.mc.get_seq_column_names_by_block_name(
                                                                                       block_name)))

        for seq_name in self.ta_seq_column_blocks:
            with tf.name_scope("{}_{}_Attention_Layer_Summary".format(self.critic_model_name, seq_name)):
                base_ops.add_norm2_summary("{}_atten_dnn_hidden_layer".format(seq_name))
                base_ops.add_dense_output_summary("{}_atten_dnn_hidden_output".format(seq_name))
                base_ops.add_weight_summary("{}_atten_dnn_hidden_layer".format(seq_name))
                base_ops.add_norm2_summary("{}_atten_dnn_hidden_layer_ffn".format(seq_name))
                base_ops.add_dense_output_summary("{}_atten_dnn_hidden_output_ffn".format(seq_name))
                base_ops.add_weight_summary("{}_atten_dnn_hidden_layer_ffn".format(seq_name))
                tf.summary.scalar(name='{}_real_len'.format(seq_name),
                                  tensor=tf.reduce_mean(self.seq_real_len[seq_name]))
                
        for seq_name in self.ta_seq_column_blocks:
            with tf.name_scope("{}_{}_Pooling_Layer_Summary".format(self.critic_model_name, seq_name)):
                base_ops.add_norm2_summary("{}_pooling_dnn_hidden_layer".format(seq_name))
                base_ops.add_dense_output_summary("{}_pooling_dnn_hidden_output".format(seq_name))
                base_ops.add_weight_summary("{}_pooling_dnn_hidden_layer".format(seq_name))