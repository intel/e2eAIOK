import logging
import os
import time
import numpy as np

import tensorflow as tf
import horovod.tensorflow as hvd
from tensorflow.python.keras import backend as K

from tf.schedulers import get_schedule

class TensorflowTrainer:
    def __init__(self, args, model):
        self.model_dir = os.path.join(args.model_dir, 'train'+str(time.time()))
        self.model = model
        self.optimizer = tf.keras.optimizers.get(args.optimizer)
        if args.amp:
            self.optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(
                self.optimizer,
                loss_scale='dynamic'
            )
        self.compiled_loss = tf.keras.losses.get(args.loss)
        self.eval_loss = tf.keras.metrics.Mean()
        self.metric = tf.keras.metrics.get(args.metric)
        self.schedule = get_schedule(args.learning_rate, args.warmup_steps)
        self.amp = args.amp
        self.num_epochs = args.num_epochs
        self.metric_threshold = args.metric_threshold

    def train(self, train_dataset, eval_dataset, metrics_print_interval=10):
        logger = logging.getLogger('upm')

        writer = tf.summary.create_file_writer(os.path.join(self.model_dir, 'event_files' + str(hvd.local_rank())))

        current_step_var = tf.Variable(0, trainable=False, dtype=tf.int64)

        checkpoint = tf.train.Checkpoint(
            optimizer=self.optimizer,
            model=self.model,
            current_step=current_step_var
        )
        manager = tf.train.CheckpointManager(
            checkpoint=checkpoint,
            directory=os.path.join(self.model_dir, 'checkpoint'),
            max_to_keep=1
        )

        @tf.function
        def train_step(x, y, first_batch):
            with tf.GradientTape(persistent=True) as tape:
                y_pred = self.model(x, training=True)
                loss = self.compiled_loss(y, y_pred)
                loss = self.optimizer.get_scaled_loss(loss) if self.amp else loss

            tape = hvd.DistributedGradientTape(tape)

            self.metric.update_state(y, y_pred)

            vars = self.model.trainable_variables
            grads = tape.gradient(loss, vars)
            if self.amp:
                grads = self.optimizer.get_unscaled_gradients(grads)

            self.optimizer.apply_gradients(zip(grads, vars))
            if first_batch:
                hvd.broadcast_variables(self.model.variables, root_rank=0)
                hvd.broadcast_variables(self.optimizer.variables(), root_rank=0)
            return loss

        @tf.function
        def evaluation_step(x, y):
            predictions = self.model(x, training=False)
            loss = self.compiled_loss(y, predictions)
            self.metric.update_state(y, predictions)
            return loss

        with writer.as_default():
            time_metric_start = time.time()
            for epoch in range(1, self.num_epochs + 1):
                for step, (x, y) in enumerate(train_dataset):
                    current_step = np.asscalar(current_step_var.numpy())
                    self.schedule(optimizer=self.optimizer, current_step=current_step)

                    self.metric.reset_states()
                    loss = train_step(x, y, epoch == 1 and step == 0)
                    if hvd.rank() == 0:
                        tf.summary.scalar(f'{self.metric.name}', self.metric.result(), step=current_step)
                        tf.summary.scalar('loss', loss, step=current_step)
                        tf.summary.scalar('schedule', K.get_value(self.optimizer.lr), step=current_step)
                        writer.flush()

                    if current_step % metrics_print_interval == 0:
                        time_metric_end = time.time()
                        train_data = {self.metric.name: f'{self.metric.result().numpy():.4f}'}
                        train_data['loss'] = f'{loss.numpy():.4f}'
                        train_data['time'] = f'{(time_metric_end - time_metric_start):.4f}'
                        logger.info(f'step: {current_step}, {train_data}')
                        time_metric_start = time.time()

                    current_step_var.assign_add(1)

                self.metric.reset_states()
                self.eval_loss.reset_states()

                for step, (x, y) in enumerate(eval_dataset):
                    loss = evaluation_step(x, y)
                    self.eval_loss.update_state(loss)

                eval_loss_reduced = hvd.allreduce(self.eval_loss.result())

                metrics_reduced = {f'{self.metric.name}_val': hvd.allreduce(self.metric.result())}            

                eval_data = {name: result.numpy() for name, result in metrics_reduced.items()}
                eval_data.update({'loss_val': eval_loss_reduced.numpy()})

                for name, result in eval_data.items():
                    tf.summary.scalar(name, result, step=current_step_var.numpy())
                writer.flush()
                logger.info(f'step: {current_step_var.numpy()}, {eval_data}')
                if hvd.rank() == 0:
                    manager.save()

                if eval_data[f'{self.metric.name}_val'] >= self.metric_threshold:
                    logger.info(f'early stop at {self.metric.name}: {eval_data[self.metric.name+"_val"]}')
                    break

            if hvd.rank() == 0:
                logger.info(f'Final eval result: {eval_data}')
                tf.saved_model.save(self.model, self.model_dir)
        return eval_data[f'{self.metric.name}_val'] 