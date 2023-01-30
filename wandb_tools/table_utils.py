from typing import List
import numpy as np


def sign2align(sign: int):
    if sign == 0:
        return '^'
    elif sign > 0:
        return '>'
    else:
        return '<'


def calc_precision(x: float):
    return int(-np.floor(np.log10(x)))


def from_with_std(mean_value: float, std_value: float,
                  width: int, decimals: int = 1,
                  precision: int = None, align: int = 0,
                  show_exp: bool = True):
    if precision is None:
        if not np.isnan(std_value):
            prec_value = std_value
        elif not np.isnan(mean_value):
            prec_value = mean_value
        else:
            prec_value = 1
        precision = calc_precision(prec_value)

    form_mean = mean_value*(10**precision)
    form_std = std_value*(10**precision)
    mean_str = '{:>.{}f}'.format(form_mean, decimals)
    std_str = '{:<.{}f}'.format(form_std, decimals)
    if show_exp:
        exp_str = 'e{}{:0>2d}'.format('-' if precision > 0 else '+', abs(precision))
    else:
        exp_str = ''
    result_str = '{}±{}{}'.format(mean_str, std_str, exp_str)
    formed_str = '{:{}{}s}'.format(result_str, sign2align(align), width)
    return formed_str


def from_wo_std(mean_value: float,
                width: int, decimals: int = 1,
                precision: int = None, align: int = 0,
                show_exp: bool = True):
    if precision is None:
        if not np.isnan(mean_value):
            prec_value = mean_value
        else:
            prec_value = 1
        precision = calc_precision(prec_value)

    form_mean = mean_value*(10**precision)
    mean_str = '{:>.{}f}'.format(form_mean, decimals)
    if show_exp:
        exp_str = 'e{}{:0>2d}'.format('-' if precision > 0 else '+', abs(precision))
    else:
        exp_str = ''
    result_str = '{}{}'.format(mean_str, exp_str)
    formed_str = '{:{}{}s}'.format(result_str, sign2align(align), width)
    return formed_str


def form_result(error_list: List[float], use_std: bool = True):
    if len(error_list) == 0:
        return '{}±{}'.format('   nan', 'nan   ')
    error_list = [e for e in error_list if not np.isnan(e)]
    error_list = np.array(error_list)
    mean_result = np.mean(error_list)
    if not use_std:
        return '{:.2e}'.format(mean_result)
    if len(error_list) == 1:
        std_result = 0.0
        prec = int(-np.floor(np.log10(mean_result)))
    else:
        std_result = np.std(error_list, ddof=1)
        prec = int(-np.floor(np.log10(std_result)))
    form_mean = mean_result*(10**prec)
    form_std = std_result*(10**prec)
    mean_str = '{:>6.{}f}'.format(form_mean, 1)
    std_str = '{:<3.{}f}'.format(form_std, 1)
    exp_str = 'e{}{:0>2d}'.format('-' if prec > 0 else '+', abs(prec))
    # return '{:e}±{:e}'.format(mean_result, std_result)
    return '{}±{}{}'.format(mean_str, std_str, exp_str)
