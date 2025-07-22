
class Reject:
    def __init__(self, instance, reject_threat, prediction_without_reject, prediction_probability):
        self.instance = instance
        self.reject_threat = reject_threat
        self.prediction_without_reject = prediction_without_reject
        self.prediction_probability = prediction_probability

    def __str__(self):
        reject_str_pres = "\n______________________________\n"
        reject_str_pres += str(self.instance) + "\n"
        reject_str_pres += self.reject_threat + "-for this instance\n"
        reject_str_pres += "\nPrediction that would have been made: " + str(self.prediction_without_reject)
        reject_str_pres += "\nPrediction Probability: " + str(self.prediction_probability)
        return reject_str_pres



class UnfairnessFlip(Reject):
    def __init__(self, instance, prediction_without_flip, prediction_probability, rule_flip_is_based_upon, sit_test_summary):
        Reject.__init__(self, instance, "Unfairness-Flip", prediction_without_flip, prediction_probability)
        self.sit_test_summary = sit_test_summary
        self.rule_reject_is_based_upon = rule_flip_is_based_upon

    def __str__(self):
        str_pres = Reject.__str__(self)
        str_pres += "\nFlip Based on this rule\n"
        str_pres += str(self.rule_reject_is_based_upon)

        if self.sit_test_summary != None:
            str_pres += "\nSituation Testing Score: " + str(self.sit_test_summary)
        return str_pres



class UnfairnessReject(Reject):
    def __init__(self, instance, prediction_without_reject, prediction_probability, rule_reject_is_based_upon, sit_test_summary):
        Reject.__init__(self, instance, "Unfairness Reject", prediction_without_reject, prediction_probability)
        self.sit_test_summary = sit_test_summary
        self.rule_reject_is_based_upon = rule_reject_is_based_upon


    def __str__(self):
        str_pres = Reject.__str__(self)
        str_pres += "\nRejection Based on this rule\n"
        str_pres += str(self.rule_reject_is_based_upon)

        if self.sit_test_summary != None:
            str_pres += "\nSituation Testing Score: " + str(self.sit_test_summary)
        return str_pres

class UncertaintyReject(Reject):

    def __init__(self, instance, prediction_without_reject,  prediction_probability):
        Reject.__init__(self, instance, "Uncertain Reject", prediction_without_reject,  prediction_probability)

    def __str__(self):
        str_pres = Reject.__str__(self)
        str_pres += "\nDecision will be deferred to human"
        return str_pres

class SchreuderReject(Reject):
    def __init__(self, instance, prediction_without_reject, prediction_probability):
        Reject.__init__(self, instance, "Schreuder Reject", prediction_without_reject,  prediction_probability)

class SchreuderFlip(Reject):
    def __init__(self, instance, prediction_without_flip, prediction_probability):
        Reject.__init__(self, instance, "Schreuder Flip", prediction_without_flip, prediction_probability)


def create_unfairness_based_flip(row):
    return UnfairnessFlip(
        instance = row['instance'],
        prediction_without_flip = row['prediction_without_reject'],
        prediction_probability = row['prediction probability'],
        rule_flip_is_based_upon = row['relevant_rule'],
        sit_test_summary = row['sit_test_info']
    )


def create_unfairness_based_reject(row):
    return UnfairnessReject(
        instance=row['instance'],
        prediction_without_reject = row['prediction_without_reject'],
        prediction_probability = row['prediction probability'],
        rule_reject_is_based_upon = row['relevant_rule'],
        sit_test_summary = row['sit_test_info']
    )

def create_uncertainty_based_reject(row):
    return UncertaintyReject(
        instance=row['instance'],
        prediction_without_reject=row['prediction_without_reject'],
        prediction_probability=row['prediction probability'],
    )

def create_simple_reject(row):
    return Reject(
        reject_threat="Trial",
        prediction_without_reject=row['prediction_without_reject'],
        prediction_probability=row['prediction probability'],
    )