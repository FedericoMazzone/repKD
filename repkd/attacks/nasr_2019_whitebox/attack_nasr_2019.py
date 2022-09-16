from repkd import utils

from .ml_privacy_meter.attack.meminf import initialize
from .ml_privacy_meter.utils.attack_data import AttackData


def attack(
    target_model,
    train_x, train_y,
    test_x, test_y,
    num_class,
    layers_to_exploit,
    gradients_to_exploit=[],
    exploit_loss=False,
    exploit_label=False,
    epochs=100,
    batch_size=10,
    attack_percentage=0.50,
    number_of_repetitions=1
):
    """
        Attack the target model. You can specify what information to exploit
        during the attack:
        - which layers output;
        - which layers gradient;
        - loss;
        - ground-truth label.
        Moreover, you can specify:
        - for how many epochs the attack model is trained for;
        - the portion of the original dataset the attacker already knows;
        - for how many times to repeat the attack model training.
        It returns the best accuracy of the attack model on the attack test
        set.
    """

    # Create attack datahandler
    train_datahandler = AttackData(
        train_x, train_y,
        test_x, test_y,
        num_class,
        attack_percentage=attack_percentage,
        batch_size=batch_size
    )

    best_accuracy = 0.0

    # Attack
    for attack_counter in range(1, number_of_repetitions + 1):

        utils.log("Starting attack attempt {}".format(attack_counter))

        attackobj = initialize(
            target_train_model=target_model,
            target_attack_model=target_model,
            train_datahandler=train_datahandler,
            attack_datahandler=None,
            layers_to_exploit=layers_to_exploit,
            gradients_to_exploit=gradients_to_exploit,
            exploit_loss=exploit_loss,
            exploit_label=exploit_label,
            epochs=epochs
        )

        attack_accuracy = attackobj.train_attack()
        best_accuracy = max(best_accuracy, attack_accuracy)

        utils.log(f"Attack attempt {attack_counter}/{number_of_repetitions}\
            finished: {attack_accuracy} ({best_accuracy})\n")

    return best_accuracy
