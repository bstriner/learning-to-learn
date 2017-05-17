from tqdm import tqdm
import csv
import os
import theano.tensor as T
import theano
import numpy as np
from .dense_layer import DenseLayer
from .util import leaky_relu, accuracy, logit, logit_np
from .mlp import MLP
from theano.tensor.shared_randomstreams import RandomStreams
from keras.optimizers import Adam


class GreedyModel(object):
    def __init__(self,
                 inner_model,
                 loss_function,
                 target_type,
                 lr_opt,
                 units=256):
        self.inner_model = inner_model
        self.loss_function = loss_function
        assert isinstance(inner_model, MLP)
        # input is log lr and log epoch
        # output is new log lr
        #inner_lr0_logit_np = np.float32(logit_np(inner_lr0))
        #logit_lr = theano.shared(value=inner_lr0_logit_np, name="lr")
        batch = theano.shared(value=np.int32(1), name="batch")

        #    (logit_lr, inner_lr0_logit_np),

        reset_vars = [
            (batch, np.int32(1))
        ]

        log_batch = T.log(batch)

        # Build LR prediction model
        lr_model = MLP([DenseLayer(1, units, activation=leaky_relu, name="outer_1"),
                        DenseLayer(units, units, activation=leaky_relu, name="outer_2"),
                        DenseLayer(units, units, activation=leaky_relu, name="outer_3"),
                        DenseLayer(units, 1, name="outer_4")])

        lr_input = T.reshape(log_batch, (1, 1))
        lr_next_logit = lr_model.call(lr_input)[0, 0]
        lr_next = T.nnet.sigmoid(lr_next_logit)
        # log_lr_next = T.clip(log_lr_next,
        #                     np.float32(np.log(lr_bounds[0])),
        #                     np.float32(np.log(lr_bounds[1])))

        # inner inputs
        input_x_train = T.fmatrix(name="input_x_train")
        target_y_train = target_type(name="target_y_train")
        input_x_val = T.fmatrix(name="input_x_val")
        target_y_val = target_type(name="target_y_val")

        # current loss
        ypred = inner_model.call(input_x_train)
        loss = loss_function(target_y_train, ypred)
        acc = accuracy(target_y_train, ypred)

        # current val loss
        ypred_val = inner_model.call(input_x_val)
        loss_val = loss_function(target_y_val, ypred_val)
        acc_val = accuracy(target_y_val, ypred_val)

        # update params
        print "LR next: {}, {}".format(lr_next.ndim, lr_next.dtype)
        inner_weights_next = [w - (lr_next * T.grad(loss, w)) for w in inner_model.weights]

        # next loss
        ypred_next = inner_model.call_on_weights(input_x_train, inner_weights_next)
        loss_next = loss_function(target_y_train, ypred_next)

        # next val loss
        ypred_next_val = inner_model.call_on_weights(input_x_val, inner_weights_next)
        loss_next_val = loss_function(target_y_val, ypred_next_val)

        # train LR model to minimize validation loss
        lr_updates = lr_opt.get_updates(lr_model.weights, {}, loss_next_val)

        inner_updates = [(w, T.cast(nw, 'float32')) for w, nw in zip(inner_model.weights, inner_weights_next)]
        other_updates = [
        #    (logit_lr, T.cast(lr_next_logit, 'float32')),
            (batch, batch + 1)
        ]
        updates = inner_updates + lr_updates + other_updates
        # for u in updates:
        #    print "Update {}: {}/{} -> {}/{}".format(u[0].name, u[0].ndim, u[0].dtype,
        #                                             u[1].ndim, u[1].dtype)
        inputs = [input_x_train, target_y_train, input_x_val, target_y_val]
        outputs = [loss, acc, loss_val, acc_val, loss_next, loss_next_val, lr_next]
        self.train_function = theano.function(inputs,
                                              outputs,
                                              updates=updates)
        srng = RandomStreams(123)
        self.reset_function = theano.function([], [], updates=reset_vars + inner_model.reset_updates(srng))

        # initialize the models before training
        print "Initializing model"
        reset_lr_function = theano.function([], [], updates=lr_model.reset_updates(srng))
        reset_lr_function()
        self.reset_function()

        self.validation_function = theano.function(inputs,
                                                   [loss, acc, loss_val, acc_val, lr_next],
                                                   updates=inner_updates + other_updates)

        lr_p = theano.shared(np.float32(0), name="initial_lr_p")
        lr_initial = T.nnet.sigmoid(lr_p)
        initial_weights = [p[1] for p in inner_model.reset_updates(srng)]
        initial_y = inner_model.call_on_weights(input_x_train, initial_weights)
        initial_loss = loss_function(target_y_train, initial_y)
        initial_y_val = inner_model.call_on_weights(input_x_val, initial_weights)
        initial_loss_val = loss_function(target_y_val, initial_y_val)
        initial_weights_next = [w - (lr_initial * T.grad(initial_loss, w)) for w in initial_weights]
        initial_y_next = inner_model.call_on_weights(input_x_train, initial_weights_next)
        initial_loss_next = loss_function(target_y_train, initial_y_next)
        initial_y_val_next = inner_model.call_on_weights(input_x_val, initial_weights_next)
        initial_loss_val_next = loss_function(target_y_val, initial_y_val_next)
        initial_opt = Adam(1e-3)
        initial_updates = initial_opt.get_updates([lr_p], {}, initial_loss_val_next)
        self.lr0_function = theano.function(inputs,
                                            [initial_loss, initial_loss_val,
                                             initial_loss_next, initial_loss_val_next,
                                             lr_initial],
                                            updates=initial_updates)

    def train_initial_lr(self, gen, epochs, output_path, verbose=True):
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        with open(output_path, 'wb') as f:
            w = csv.writer(f)
            w.writerow(["Epoch", "Loss", "Validation Loss", "Next Loss", "Next Validation Loss", "LR"])
            for epoch in tqdm(range(epochs), desc="Training initial LR"):
                inputs = next(gen)
                losses = self.lr0_function(*inputs)
                w.writerow([epoch] + losses)
                if verbose:
                    tqdm.write("Epoch {}: Loss: {}, Val Loss: {}, LR: {}".format(epoch,
                                                                                 losses[2], losses[3], losses[4]))

    def validate(self, gen, batches, validation_epochs, output_path):
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        epoch_data = []
        for epoch in tqdm(range(validation_epochs), desc="Validation"):
            data = []
            self.reset_function()
            for batch in tqdm(range(batches), desc="Validation Epoch {}".format(epoch)):
                inputs = next(gen)
                outputs = self.validation_function(*inputs)
                data.append(outputs)
            epoch_data.append(data)
        with open(output_path, 'wb') as f:
            w = csv.writer(f)
            w.writerow(['Batch'] +
                       ['Loss {}'.format(i) for i in range(validation_epochs)] +
                       ['Accuracy {}'.format(i) for i in range(validation_epochs)] +
                       ['Validation Loss {}'.format(i) for i in range(validation_epochs)] +
                       ['Validation Accuracy {}'.format(i) for i in range(validation_epochs)] +
                       ['LR {}'.format(i) for i in range(validation_epochs)] +
                       ["Average Loss", "Average Accuracy",
                        "Average Validation Loss", "Average Validation Accuracy",
                        "Average LR"])
            for batch in range(batches):
                avgs = [np.mean([epoch_data[e][batch][metric]
                                 for e in range(validation_epochs)])
                        for metric in range(5)]
                w.writerow([batch] +
                           [epoch_data[i][batch][j] for j in range(5) for i in range(validation_epochs)] +
                           avgs)

    def train(self, gen, epochs, batches, validation_epochs, frequency, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        self.validate(gen=gen, batches=batches, validation_epochs=validation_epochs,
                      output_path=os.path.join(output_path, "initial.csv"))

        with open(os.path.join(output_path, "history.csv"), 'wb') as f:
            w = csv.writer(f)
            w.writerow(["Epoch",
                        "Final Train Loss",
                        "Final Train Acc",
                        "Final Validation Loss",
                        "Final Validation Acc",
                        "Final Learning Rate"])
            for epoch in tqdm(range(epochs), desc="Training"):
                # Reset MLP weights
                self.reset_function()
                for batch in tqdm(range(batches), desc="Epoch {}".format(epoch)):
                    inputs = next(gen)
                    losses = self.train_function(*inputs)
                w.writerow([epoch, losses[0], losses[1], losses[2], losses[3], losses[6]])

                if (epoch + 1) % frequency == 0:
                    self.validate(gen=gen, batches=batches, validation_epochs=validation_epochs,
                                  output_path=os.path.join(output_path, "epoch-{:09d}.csv".format(epoch)))
