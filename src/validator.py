
import numpy as np

class Validator():
  def __init__(self):
    self.score = np.inf

  def new_try(self,new_score):
    if new_score < self.score:
      self.score = new_score
      return True
    return False






