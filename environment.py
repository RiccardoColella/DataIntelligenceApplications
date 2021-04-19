import numpy as np


class Environment:
    def __init__(self):
        product_price=10
        self.product_price = product_price
        self.customer_class_1 = Customer(a_new_users=-5, b_new_users=1, c_new_users=1, d_new_users=100, var_new_users=2,
                                         a_cost_per_click=0.9,
                                         a_conversion_rate=-1, b_conversion_rate=20, c_conversion_rate=0.7,
                                         price_min=product_price,
                                         mean_n_times_comeback=5, dev_n_times_comeback=0.2)
        self.customer_class_2 = Customer(a_new_users=-5, b_new_users=0, c_new_users=2, d_new_users=75, var_new_users=2,
                                         a_cost_per_click=0.8,
                                         a_conversion_rate=-1, b_conversion_rate=20, c_conversion_rate=0.9,
                                         price_min=product_price,
                                         mean_n_times_comeback=10, dev_n_times_comeback=0.5)
        self.customer_class_3 = Customer(a_new_users=-4, b_new_users=0, c_new_users=1, d_new_users=100, var_new_users=2,
                                         a_cost_per_click=0.95,
                                         a_conversion_rate=-1, b_conversion_rate=50, c_conversion_rate=0.5,
                                         price_min=product_price,
                                         mean_n_times_comeback=15, dev_n_times_comeback=0.2)

        self.customer_classes = [self.customer_class_1, self.customer_class_2, self.customer_class_3]

    def get_margin(self, selling_price):
        return selling_price - self.product_price

    def get_all_new_users_daily(self,bid):
        """
        Given a bid this function returns the number of new users for each class
        :param bid: the proposed bid
        :return: a triplet with (n_users_class1, n_users_class2, n_users_class3)
        """
        new_user_1 = self.get_new_users_daily(bid, 1)
        new_user_2 = self.get_new_users_daily(bid, 2)
        new_user_3 = self.get_new_users_daily(bid, 3)

        return new_user_1, new_user_2, new_user_3

    def get_cost_per_click(self, bid, chosen_class):
        """
        Given a bid, this function returns the cost of every click for the given customer class
        :param bid: the proposed class
        :param chosen_class: the requested customer class
        :return: the cost of every click for the given customer class
        """
        return self.customer_classes[chosen_class - 1].cost_per_click_daily(bid)

    def get_all_cost_per_click(self,bid):
        """
        Given a bid, this function returns the cost of every click for every customer in each category
        :param bid: the proposed bid
        :return: a triplet containing the cost of each click for evey customer class
        """
        cost1 = self.get_cost_per_click(bid, 1)
        cost2 = self.get_cost_per_click(bid, 2)
        cost3 = self.get_cost_per_click(bid, 3)
        return cost1, cost2, cost3

    def get_conversion_rate(self, price, chosen_class):
        """
        Given a class, returns it's conversion rate
        :param price: price for which calculate the conversion rate
        :param chosen_class: the class for which calculate the conversion rate
        :return: a percentage that indicates the conversion rate
        """
        return self.customer_classes[chosen_class - 1].conversion_rate(price)

    def buy(self, price, chosen_class):
        """
        This function tells if the user will buy or not the item
        :param price: price of selling
        :param chosen_class: class the user belongs
        :return: 1 if buys, 0 if not
        """
        return self.customer_classes[chosen_class - 1].buy(price)

    def n_time_comeback(self, chosen_class):
        """
        This function returns a stocastic number of how many times the user will come back
        :param chosen_class: class the user belongs
        :return: number of times the user will come back in the following 30 days
        """
        return self.customer_classes[chosen_class].n_times_comeback()

    def get_new_users_daily(self, bid, chosen_class):
        return self.customer_classes[chosen_class - 1].new_users_daily_clicks(bid)

    def get_mean_n_times_comeback(self, chosen_class):
        return self.customer_classes[chosen_class - 1].get_mean_n_times_comeback()


class Customer:
    """ This class represents a customer class.
    Three items of this class will be instantiated: C1, C2, C3.

    Attributes:
        seed    Example attribute
    """

    def __init__(self,
                 a_new_users, b_new_users, c_new_users, d_new_users, var_new_users,
                 a_cost_per_click,
                 a_conversion_rate, b_conversion_rate, c_conversion_rate, price_min,
                 mean_n_times_comeback, dev_n_times_comeback):
        """

        :param a_new_users: bid coeff
        :param b_new_users: bid ** 2 coeff
        :param c_new_users: bid ** 3 coeff
        :param d_new_users: const  multiplier of all function
        :param a_cost_per_click: coefficient of bid's value
        :param a_conversion_rate: coeff of (price - price_min)
        :param b_conversion_rate: denominator of exponent of (price - price_min)
        :param c_conversion_rate: const multiplier of the whole equation
        :param mean_n_times_comeback: mean of the normal distribution
        :param dev_n_times_comeback: standard deviation of the normal distribution
        """
        self.a_new_users = a_new_users
        self.b_new_users = b_new_users
        self.c_new_users = c_new_users
        self.d_new_users = d_new_users
        self.var_new_users = var_new_users

        self.a_cost_per_click = a_cost_per_click

        self.a_conversion_rate = a_conversion_rate
        self.b_conversion_rate = b_conversion_rate
        self.c_conversion_rate = c_conversion_rate
        self.price_min = price_min

        self.mean_n_times_comeback = mean_n_times_comeback
        self.dev_n_times_comeback = dev_n_times_comeback

    def new_users_daily_clicks_mean(self, bid):
        a = self.a_new_users
        b = self.b_new_users
        c = self.c_new_users
        d = self.d_new_users
        return d * (1.0 - np.exp(a * bid + b * bid ** 2 + c * bid ** 3))

    def new_users_daily_clicks(self, bid):
        """ Customer's characteristic n.1.
        This method returns a stochastic number of daily clicks of new users (i.e., that have never clicked before these
        ads) as a function depending on the bid.
        :param bid: The seller's bid
        :return: Number of new users that clicks on the ads in a day
        """
        # Functions that assign the number of clicks to a given budget
        # They are monotone increasing in [0,1]
        a = self.a_new_users
        b = self.b_new_users
        c = self.c_new_users
        d = self.d_new_users
        mean = self.new_users_daily_clicks_mean(bid)
        var = self.var_new_users
        return round(np.random.normal(loc=mean, scale=var))

    def cost_per_click_daily(self, bid):
        """ Customer's characteristic n. 2.
        This method returns a daily stochastic cost per click as a function of the bid.
        The seller pays more for this class depending on his bid. What the seller pays is not what he bids: here we must
        model the second price auction without having the other bids.
        :param bid: The seller's bid
        :return: The cost that will be due for the customer of this customer with the given bid
        """
        return self.a_cost_per_click * bid

    def conversion_rate(self, price):
        """ Customer's characteristic n. 3.
        This method is a conversion rate function. It provids the probability that a user will buy the item given a
        price.
        :param price: The item's price
        :param price_min: The item's min price possible
        :return: The probability that a user will buy the item given a price
        """
        a = self.a_conversion_rate
        b = self.b_conversion_rate
        c = self.c_conversion_rate
        price_min = self.price_min
        # Probabilities of conversion given a price
        return c * np.exp(a * (price - price_min) ** (1 / 2) / b)

    def buy(self, price):
        """
        This method returns if the customer will buy or not the item at the given price
        :param price: given price
        :return: 1 if the customer will buy they item. 0 otherwise.
        """
        return np.random.binomial(1, self.conversion_rate(price))

    def n_times_comeback(self):
        """ Custemer's characteristic n. 4.
        This method computes the distribution probability over the number of times the user will come back to the
        ecommerce website to buy that item by 30 days after the first purchase (and simulate such visits in future).
        :return: The probability that the user of this class will come back the given number of times
        """
        # OR Poisson or Gaussian
        loc = self.get_mean_n_times_comeback()
        scale = self.dev_n_times_comeback
        return round(np.random.normal(loc=loc, scale=scale, size=1))

    def get_mean_n_times_comeback(self):
        return self.mean_n_times_comeback
