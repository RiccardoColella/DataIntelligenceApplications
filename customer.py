class Customer:
    """ This class represents a customer class.
    Three items of this class will be instantiated: C1, C2, C3.

    Attributes:
        seed    Example attribute,
    """

    seed = 0

    def __init__(self, seed):
        self.seed = 0

    def new_users_daily_clicks(self, bid):
        """ Customer's characteristic n.1.
        This method returns a stochastic number of daily clicks of new users (i.e., that have never clicked before these
        ads) as a function depending on the bid.
        :param bid: The seller's bid
        :return: Number of new users that clicks on the ads in a day
        """
        # todo: write the method
        return (bid + self.seed)/2

    def cost_per_click_daily(self, bid):
        """ Customer's characteristic n. 2.
        This method returns a daily stochastic cost per click as a function of the bid.
        :param bid: The seller's bid
        :return:
        """
        # todo: write the method. Also: understand the purpose

    def coversion_rate(self, price):
        """ Customer's characteristic n. 3.
        This method is a conversion rate function. It provids the probability that a user will buy the item given a
        price.
        :param price: The item's price
        :return: The probability that a user will buy the item given a price
        """
        # todo: write the method

    def comeback_probability(self, n_times):
        """ Custemer's characteristic n. 4.
        This method computes the distribution probability over the number of times the user will come back to the
        ecommerce website to buy that item by 30 days after the first purchase (and simulate such visits in future).
        :param n_times: The number of times over which computer the probability for the customer to comeback
        :return: The probability that the user of this class will come back the given number of times
        """
        # todo: write the method
