'''
The Attack class.
'''

import numpy as np
import tensorflow as tf
from repkd import utils

from ..utils.attack_utils import attack_utils, sanity_check
from ..utils.losses import CrossEntropyLoss, mse
from ..utils.optimizers import optimizer_op
from .meminf_modules.create_cnn import (cnn_for_cnn_gradients,
                                        cnn_for_cnn_layeroutputs,
                                        cnn_for_fcn_gradients)
from .meminf_modules.create_fcn import fcn_module
from .meminf_modules.encoder import create_encoder

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


# Sets soft placement below for GPU memory issues
tf.config.set_soft_device_placement(True)

ioldinit = tf.compat.v1.Session.__init__


def myinit(session_object, target='', graph=None, config=None):
    config = tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=True)


tf.compat.v1.Session.__init__ = myinit


# To decide what attack component (FCN or CNN) to
# use on the basis of the layer name.
# CNN_COMPONENTS_LIST are the layers requiring each input in 3 dimensions.
# GRAD_COMPONENTS_LIST are the layers which have trainable components for computing gradients
CNN_COMPONENT_LIST = ['Conv', 'MaxPool']
GRAD_LAYERS_LIST = ['Conv', 'Dense']


class initialize(object):
    """
    This attack was originally proposed by Nasr et al. It exploits
    one-hot encoding of true labels, loss value, intermediate layer 
    activations and gradients of intermediate layers of the target model 
    on data points, for training the attack model to infer membership 
    in training data.

    Paper link: https://arxiv.org/abs/1812.00910

    Args:
    ------
    target_train_model: The target (classification) model that'll 
                        be used to train the attack model.

    target_attack_model: The target (classification) model that we are
                         interested in quantifying the privacy risk of. 
                         The trained attack model will be used 
                         for attacking this model to quantify its membership
                         privacy leakage. 

    train_datahandler: an instance of `ml_privacy_meter.data.attack_data.load`,
                       that is used to retrieve dataset for training the 
                       attack model. The member set of this training set is
                       a subset of the classification model's
                       training set. Check Main README on how to 
                       load dataset for the attack.

    attack_datahandler: an instance of `ml_privacy_meter.data.attack_data.load`,
                        used to retrieve dataset for evaluating the attack 
                        model. The member set of this test/evaluation set is
                        a subset of the target attack model's train set minus
                        the training members of the target_train_model.

    optimizer: The optimizer op for training the attack model.
               Default op is "adam".

    layers_to_exploit: a list of integers specifying the indices of layers,
                       whose activations will be exploited by the attack model.
                       If the list has only a single element and 
                       it is equal to the index of last layer,
                       the attack can be considered as a "blackbox" attack.

    gradients_to_exploit: a list of integers specifying the indices 
                          of layers whose gradients will be 
                          exploited by the attack model. 

    exploit_loss: boolean; whether to exploit loss value of target model or not.

    exploit_label: boolean; whether to exploit one-hot encoded labels or not.                 

    learning_rate: learning rate for training the attack model 

    epochs: Number of epochs to train the attack model 

    Examples:
    """

    def __init__(self,
                 target_train_model,
                 target_attack_model,
                 train_datahandler,
                 attack_datahandler,
                 device=None,
                 optimizer="adam",
                 model_name="sample_model",
                 layers_to_exploit=None,
                 gradients_to_exploit=None,
                 exploit_loss=True,
                 exploit_label=True,
                 learning_rate=0.001,
                 epochs=100):

        self.attack_utils = attack_utils()
        self.target_train_model = target_train_model
        self.target_attack_model = target_attack_model
        self.train_datahandler = train_datahandler
        self.attack_datahandler = attack_datahandler
        self.optimizer = optimizer_op(optimizer, learning_rate)
        self.layers_to_exploit = layers_to_exploit
        self.gradients_to_exploit = gradients_to_exploit
        self.exploit_loss = exploit_loss
        self.device = device
        self.exploit_label = exploit_label
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.output_size = int(target_train_model.output.shape[1])
        self.ohencoding = self.attack_utils.createOHE(self.output_size)
        self.model_name = model_name

        # Create input containers for attack & encoder model.
        self.create_input_containers()
        layers = target_train_model.layers

        # basic sanity checks
        sanity_check(layers, layers_to_exploit)
        sanity_check(layers, gradients_to_exploit)

        # Create individual attack components
        self.create_attack_components(layers)

        # Initialize the attack model
        self.initialize_attack_model()

    def create_input_containers(self):
        """
        Creates arrays for inputs to the attack and 
        encoder model. 
        (NOTE: Although the encoder is a part of the attack model, 
        two sets of containers are required for connecting 
        the TensorFlow graph).
        """
        self.attackinputs = []
        self.encoderinputs = []

    def create_layer_components(self, layers):
        """
        Creates CNN or FCN components for layers to exploit
        """
        for l in self.layers_to_exploit:
            # For each layer to exploit, module created and added to self.attackinputs and self.encoderinputs
            layer = layers[l-1]
            input_shape = layer.output_shape[1]
            requires_cnn = map(lambda i: i in layer.__class__.__name__,
                               CNN_COMPONENT_LIST)
            if any(requires_cnn):
                module = cnn_for_cnn_layeroutputs(layer.output_shape)
            else:
                module = fcn_module(input_shape, 100)
            self.attackinputs.append(module.input)
            self.encoderinputs.append(module.output)

    def create_label_component(self, output_size):
        """
        Creates component if OHE label is to be exploited
        """
        module = fcn_module(output_size)
        self.attackinputs.append(module.input)
        self.encoderinputs.append(module.output)

    def create_loss_component(self):
        """
        Creates component if loss value is to be exploited
        """
        module = fcn_module(1, 100)
        self.attackinputs.append(module.input)
        self.encoderinputs.append(module.output)

    def create_gradient_components(self, model, layers):
        """
        Creates CNN/FCN component for gradient values of layers of gradients to exploit
        """
        grad_layers = []
        for layer in layers:
            if any(map(lambda i: i in layer.__class__.__name__, GRAD_LAYERS_LIST)):
                grad_layers.append(layer)
        variables = model.variables
        for layerindex in self.gradients_to_exploit:
            # For each gradient to exploit, module created and added to self.attackinputs and self.encoderinputs
            layer = grad_layers[layerindex-1]
            shape = self.attack_utils.get_gradshape(variables, layerindex)
            requires_cnn = map(lambda i: i in layer.__class__.__name__,
                               CNN_COMPONENT_LIST)
            if any(requires_cnn):
                module = cnn_for_cnn_gradients(shape)
            else:
                module = cnn_for_fcn_gradients(shape)
            self.attackinputs.append(module.input)
            self.encoderinputs.append(module.output)

    def create_attack_components(self, layers):
        """
        Creates FCN and CNN modules constituting the attack model.  
        """
        model = self.target_train_model

        # for layer outputs
        if self.layers_to_exploit and len(self.layers_to_exploit):
            self.create_layer_components(layers)

        # for one hot encoded labels
        if self.exploit_label:
            self.create_label_component(self.output_size)

        # for loss
        if self.exploit_loss:
            self.create_loss_component()

        # for gradients
        if self.gradients_to_exploit and len(self.gradients_to_exploit):
            self.create_gradient_components(model, layers)

        # encoder module
        self.encoder = create_encoder(self.encoderinputs)

    def initialize_attack_model(self):
        """
        Initializes a `tf.keras.Model` object for attack model.
        The output of the attack is the output of the encoder module.
        """
        output = self.encoder
        self.attackmodel = tf.compat.v1.keras.Model(inputs=self.attackinputs,
                                                    outputs=output)

    def get_layer_outputs(self, model, features):
        """
        Get the intermediate computations (activations) of 
        the hidden layers of the given target model.
        """
        layers = model.layers
        for l in self.layers_to_exploit:
            x = model.input
            y = layers[l-1].output
            # Model created to get output of specified layer
            new_model = tf.compat.v1.keras.Model(x, y)
            predicted = new_model(features)
            self.inputArray.append(predicted)

    def get_labels(self, labels):
        """
        Retrieves the one-hot encoding of the given labels.
        """
        ohe_labels = self.attack_utils.one_hot_encoding(
            labels, self.ohencoding)
        return ohe_labels

    def get_loss(self, model, features, labels):
        """
        Computes the loss for given model on given features and labels
        """
        logits = model(features)
        loss = CrossEntropyLoss(logits, labels)

        return loss

    def compute_gradients(self, model, features, labels):
        """
        Computes gradients given the features and labels using the loss
        """
        split_features = self.attack_utils.split(features)
        split_labels = self.attack_utils.split(labels)
        gradient_arr = []
        for (feature, label) in zip(split_features, split_labels):
            with tf.GradientTape() as tape:
                logits = model(feature)
                loss = CrossEntropyLoss(logits, label)
            targetvars = model.variables
            grads = tape.gradient(loss, targetvars)
            # Add gradient wrt crossentropy loss to gradient_arr
            gradient_arr.append(grads)

        return gradient_arr

    def get_gradients(self, model, features, labels):
        """
        Retrieves the gradients for each example.
        """
        gradient_arr = self.compute_gradients(model, features, labels)
        batch_gradients = []
        for grads in gradient_arr:
            # gradient_arr is a list of size of number of layers having trainable parameters
            gradients_per_example = []
            for g in self.gradients_to_exploit:
                g = (g-1)*2
                shape = grads[g].shape
                reshaped = (int(shape[0]), int(shape[1]), 1)
                toappend = tf.reshape(grads[g], reshaped)
                gradients_per_example.append(toappend.numpy())
            batch_gradients.append(gradients_per_example)

        # Adding the gradient matrices fo batches
        batch_gradients = np.asarray(batch_gradients)
        splitted = np.hsplit(batch_gradients, batch_gradients.shape[1])
        for s in splitted:
            array = []
            for i in range(len(s)):
                array.append(s[i][0])
            array = np.asarray(array)

            self.inputArray.append(array)

    def get_gradient_norms(self, model, features, labels):
        """
        Retrieves the gradients for each example
        """
        gradient_arr = self.compute_gradients(model, features, labels)
        batch_gradients = []
        for grads in gradient_arr:
            batch_gradients.append(np.linalg.norm(grads[-1]))
        return batch_gradients

    def forward_pass(self, model, features, labels):
        """
        Computes and collects necessary inputs for attack model
        """
        # container to extract and collect inputs from target model
        self.inputArray = []

        # Getting the intermediate layer computations
        if self.layers_to_exploit and len(self.layers_to_exploit):
            self.get_layer_outputs(model, features)

        # Getting the one-hot-encoded labels
        if self.exploit_label:
            ohelabels = self.get_labels(labels)
            self.inputArray.append(ohelabels)

        # Getting the loss value
        if self.exploit_loss:
            loss = self.get_loss(model, features, labels)
            loss = tf.reshape(loss, (len(loss.numpy()), 1))
            self.inputArray.append(loss)

        # Getting the gradients
        if self.gradients_to_exploit and len(self.gradients_to_exploit):
            self.get_gradients(model, features, labels)

        attack_outputs = self.attackmodel(self.inputArray)
        return attack_outputs

    def attack_accuracy(self, members, nonmembers):
        """
        Computes attack accuracy of the attack model.
        """
        attack_acc = tf.keras.metrics.Accuracy(
            'attack_acc', dtype=tf.float32)
        model = self.target_train_model

        for (membatch, nonmembatch) in zip(members, nonmembers):
            mfeatures, mlabels = membatch
            nmfeatures, nmlabels = nonmembatch

            # Computing the membership probabilities
            mprobs = self.forward_pass(model, mfeatures, mlabels)
            nonmprobs = self.forward_pass(model, nmfeatures, nmlabels)
            probs = tf.concat((mprobs, nonmprobs), 0)

            target_ones = tf.ones(mprobs.shape, dtype=bool)
            target_zeros = tf.zeros(nonmprobs.shape, dtype=bool)
            target = tf.concat((target_ones, target_zeros), 0)

            attack_acc(probs > 0.5, target)

        result = attack_acc.result()
        return result

    def compute_metrics(self):

        model = self.target_train_model

        global_accuracy_score = tf.keras.metrics.Accuracy()
        global_precision_score = tf.keras.metrics.Precision()
        global_recall_score = tf.keras.metrics.Recall()

        for i in range(self.train_datahandler.num_class):

            accuracy_score = tf.keras.metrics.Accuracy()
            precision_score = tf.keras.metrics.Precision()
            recall_score = tf.keras.metrics.Recall()

            for membatch in self.train_datahandler.mtest_byclass[i]:
                mfeatures, mlabels = membatch
                mprobs = self.forward_pass(model, mfeatures, mlabels)
                target = tf.ones(mprobs.shape, dtype=bool)
                accuracy_score(mprobs > 0.5, target)
                precision_score(mprobs > 0.5, target)
                recall_score(mprobs > 0.5, target)
                global_accuracy_score(mprobs > 0.5, target)
                global_precision_score(mprobs > 0.5, target)
                global_recall_score(mprobs > 0.5, target)

            for nonmembatch in self.train_datahandler.nmtest_byclass[i]:
                nmfeatures, nmlabels = nonmembatch
                nonmprobs = self.forward_pass(model, nmfeatures, nmlabels)
                target = tf.zeros(nonmprobs.shape, dtype=bool)
                accuracy_score(nonmprobs > 0.5, target)
                precision_score(nonmprobs > 0.5, target)
                recall_score(nonmprobs > 0.5, target)
                global_accuracy_score(nonmprobs > 0.5, target)
                global_precision_score(nonmprobs > 0.5, target)
                global_recall_score(nonmprobs > 0.5, target)

            accuracy = accuracy_score.result()
            precision = precision_score.result()
            recall = recall_score.result()

            utils.log(f"Class {i}: accuracy {accuracy:.2f}, precision "
                      f"{precision:.2f}, recall {recall:.2f}")

        accuracy = global_accuracy_score.result()
        precision = global_precision_score.result()
        recall = global_recall_score.result()

        utils.log(f"Global : accuracy {accuracy:.2f}, precision "
                  f"{precision:.2f}, recall {recall:.2f}")

    def train_attack(self):
        """
        Trains the attack model
        """
        assert self.attackmodel, "Attack model not initialized"
        model = self.target_train_model
        attack_acc = tf.keras.metrics.Accuracy('attack_acc', dtype=tf.float32)

        with tf.device(self.device):
            best_accuracy = 0.5
            for e in range(1, self.epochs + 1):
                zipped = zip(self.train_datahandler.mtrain,
                             self.train_datahandler.nmtrain)
                for((mfeatures, mlabels), (nmfeatures, nmlabels)) in zipped:
                    with tf.GradientTape() as tape:
                        tape.reset()
                        # Getting outputs of forward pass of attack model
                        moutputs = self.forward_pass(model, mfeatures, mlabels)
                        nmoutputs = self.forward_pass(
                            model, nmfeatures, nmlabels)
                        # Computing the true values for loss function according
                        memtrue = tf.ones(moutputs.shape)
                        nonmemtrue = tf.zeros(nmoutputs.shape)
                        target = tf.concat((memtrue, nonmemtrue), 0)
                        probs = tf.concat((moutputs, nmoutputs), 0)
                        attackloss = mse(target, probs)
                    # Computing gradients

                    grads = tape.gradient(
                        attackloss, self.attackmodel.variables)
                    self.optimizer.apply_gradients(
                        zip(grads, self.attackmodel.variables))

                # Calculating Attack accuracy
                attack_acc(probs > 0.5, target)

                attack_train_accuracy = self.attack_accuracy(
                    self.train_datahandler.mtrain, self.train_datahandler.nmtrain)
                attack_accuracy = self.attack_accuracy(
                    self.train_datahandler.mtest, self.train_datahandler.nmtest)

                new_bestacc = False
                if attack_accuracy > best_accuracy:
                    new_bestacc = True
                    best_accuracy = attack_accuracy

                utils.log("Epoch {} over: train acc {}, test acc {}, best test acc {}"
                          .format(e, attack_train_accuracy, attack_accuracy, best_accuracy))

                if (self.train_datahandler.num_class is not None) and new_bestacc:
                    self.compute_metrics()

        if type(best_accuracy) is float:
            return best_accuracy
        else:
            return float(best_accuracy.numpy())
