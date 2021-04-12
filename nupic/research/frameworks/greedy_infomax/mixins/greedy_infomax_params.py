# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

class GreedyInfoMaxParams(object):

    def setup_experiment(self, config):
        super().setup_experiment(config)
        params_dict = config.get("greedy_infomax_params", dict())


    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        eo["update_config_with_suggestion"].append(
            "OneCycleLRParams.update_config_with_suggestion"
        )
        return eo
