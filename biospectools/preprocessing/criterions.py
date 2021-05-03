import numpy as np


class BaseStopCriterion:
    def __init__(self, max_iter):
        self.max_iter = max_iter

        self.best_idx = 0
        self.scores = []
        self.values = []

    def add(self, score, value=None):
        self.scores.append(score)
        self.values.append(value)

        self._update_best(score)

    def _update_best(self, score):
        if score < self.scores[self.best_idx]:
            self.best_idx = len(self.scores) - 1

    def __bool__(self):
        return self.cur_iter == self.max_iter or bool(self._stop())

    @property
    def cur_iter(self):
        return len(self.scores)

    @property
    def best_score(self):
        return self.scores[self.best_idx]

    @property
    def best_value(self):
        return self.values[self.best_idx]

    @property
    def best_iter(self):
        if self.best_idx >= 0:
            return self.best_idx + 1
        else:
            return len(self.scores) - self.best_idx + 1

    def _stop(self) -> bool:
        return False


class MatlabStopCriterion(BaseStopCriterion):
    def __init__(self, max_iter, precision=None):
        super().__init__(max_iter)
        self.precision = precision
        self._eps = np.finfo(float).eps

    def _stop(self) -> bool:
        if self.cur_iter <= 2:
            return False

        pp_rmse, p_rmse, rmse = [round(r, self.precision)
                                 for r in self.scores[-3:]]

        return rmse > p_rmse or pp_rmse - rmse <= self._eps


class TolStopCriterion(BaseStopCriterion):
    def __init__(self, max_iter, tol, patience):
        super().__init__(max_iter)
        self.tol = tol
        self.patience = patience

    def _update_best(self, score):
        if score + self.tol < self.best_score:
            self.best_idx = len(self.scores) - 1

    def _stop(self) -> bool:
        if self.cur_iter <= self.patience + 1:
            return False

        no_update_iters = self.cur_iter - self.best_iter
        return no_update_iters > self.patience
