import os
import ast
import json
import matplotlib.pyplot as plt
import numpy as np

path = 'experiments/adv_tmp_JET/acc_and_loss_exp_adv_tmp_JET.txt'

loss = []
out_states_loss = []
domain_classifier_loss = []
out_states_categorical_accuracy = []
domain_classifier_categorical_accuracy = []
val_loss = []
val_out_states_loss = []
val_domain_classifier_loss = []
val_out_states_categorical_accuracy = []
val_domain_classifier_categorical_accuracy = []

with open(path, 'r') as file:
    count = 0
    while True:
        count += 1

        # Get next line from file
        line = file.readline()

        # if line is empty
        # end of file is reached
        if not line:
            break
        #print("Line{}: {}".format(count, line.strip()))
        a = ast.literal_eval(line)
        loss.append(a['loss'])
        out_states_loss.append(a['out_states_loss'])
        domain_classifier_loss.append(a['domain_classifier_loss'])
        out_states_categorical_accuracy.append(a['out_states_categorical_accuracy'])
        domain_classifier_categorical_accuracy.append(a['domain_classifier_categorical_accuracy'])
        val_loss.append(a['val_loss'])
        val_out_states_loss.append(a['val_out_states_loss'])
        val_domain_classifier_loss.append(a['val_domain_classifier_loss'])
        val_out_states_categorical_accuracy.append(a['val_out_states_categorical_accuracy'])
        val_domain_classifier_categorical_accuracy.append(a['val_domain_classifier_categorical_accuracy'])

epochs = np.arange(0, len(loss), 1)
plt.plot(epochs, loss, color='blue')
plt.plot(epochs, val_loss, color='blue', linestyle='dashed')
plt.plot(epochs, out_states_loss, color='red')
plt.plot(epochs, val_out_states_loss, color='red', linestyle='dashed')
plt.plot(epochs, domain_classifier_loss, color='green')
plt.plot(epochs, val_domain_classifier_loss, color='green', linestyle='dashed')

plt.yticks(np.arange(0, 1, 0.1))

plt.xlim([-1, 110])
#plt.ylim([0, 1])

plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(('loss', 'val loss', 'clf loss', 'val clf loss', 'dom loss', 'val dom loss'), loc = 'best')

plt.savefig('CNNLSTM_model_loss_vs_epochs_adv_tmp_JET.png')
