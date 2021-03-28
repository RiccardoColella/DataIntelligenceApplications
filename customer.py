import random
import numpy as np


class Customer:
    """ This class represents a customer class.
    Three items of this class will be instantiated: C1, C2, C3.

    Attributes:
        seed    Example attribute,
    """

    binary_feature1 = 0
    binary_feature2 = 0

    def __init__(self, binary_feature1, binary_feature2):
        self.binary_feature1 = binary_feature1
        self.binary_feature2 = binary_feature2

    def new_users_daily_clicks(self, bid, a=1, b=1, c=1, d=1):
        """ Customer's characteristic n.1.
        This method returns a stochastic number of daily clicks of new users (i.e., that have never clicked before these
        ads) as a function depending on the bid.
        :param bid: The seller's bid
        :param a: bid coeff
        :param b: bid ** 2 coeff
        :param c: bid ** 3 coeff
        :param d: const  multiplier of all function
        :return: Number of new users that clicks on the ads in a day
        """
        # Functions that assign the number of clicks to a given budget
        # They are monotone increasing in [0,1]
        return d * (1.0 - np.exp(a * bid + b * bid ** 2 + c * bid ** 3))

    def cost_per_click_daily(self, bid):
        """ Customer's characteristic n. 2.
        This method returns a daily stochastic cost per click as a function of the bid.
        The seller pays more for this class depending on his bid. What the seller pays is not what he bids: here we must
        model the second price auction without having the other bids.
        :param bid: The seller's bid
        :return:
        """
        # todo: write the method. Also: understand the purpose
        return random.random * bid

    def coversion_rate(self, price, price_min, a=-1, b=1, c=1.0):
        """ Customer's characteristic n. 3.
        This method is a conversion rate function. It provids the probability that a user will buy the item given a
        price.
        :param price: The item's price
        :param price_min: The item's min price possible
        :param a: coeff of (price - price_min)
        :param b: denominator of exponent of (price - price_min)
        :param c: const multiplier of the whole equation
        :return: The probability that a user will buy the item given a price
        """
        # Probabilities of conversion given a price
        return c * np.exp(a * (price - price_min) ** (1 / 2) / b)

    def comeback_probability(self, n_times):
        """ Custemer's characteristic n. 4.
        This method computes the distribution probability over the number of times the user will come back to the
        ecommerce website to buy that item by 30 days after the first purchase (and simulate such visits in future).
        :param n_times: The number of times over which computer the probability for the customer to comeback
        :return: The probability that the user of this class will come back the given number of times
        """
        # todo: write the method


class C1(Customer):
    def __init__(self, binary_feature1, binary_feature2):
        super().__init__(binary_feature1, binary_feature2)

    def new_users_daily_clicks(self, bid, a=-5, b=1, c=1, d=100):
        return super().new_users_daily_clicks(bid, a=a, d=d)

    def conversion_rate(self, price, price_min, a=-1, b=20, c=0.7):
        return super().coversion_rate(price, price_min, b=b, c=c)


class C2(Customer):
    def __init__(self, binary_feature1, binary_feature2):
        super().__init__(binary_feature1, binary_feature2)

    def new_users_daily_clicks(self, bid, a=-5, b=0, c=2, d=75):
        return super().new_users_daily_clicks(bid, a=a, d=d)

    def conversion_rate(self, price, price_min, a=-1, b=20, c=0.9):
        return super().coversion_rate(price, price_min, b=b, c=c)


class C3(Customer):
    def __init__(self, binary_feature1, binary_feature2):
        super().__init__(binary_feature1, binary_feature2)

    def new_users_daily_clicks(self, bid, a=-4, b=0, c=1, d=100):
        return super().new_users_daily_clicks(bid, a=a, d=d)

    def conversion_rate(self, price, price_min, a=-1, b=50, c=0.5):
        return super().coversion_rate(price*price*price, price_min, b=b, c=c)