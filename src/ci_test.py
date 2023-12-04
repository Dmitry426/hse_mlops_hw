class BaseTestClass:
    @classmethod
    def get_tests(cls):
        return [
            getattr(cls, name)
            for name in dir(cls)
            if name.startswith("test") and callable(getattr(cls, name))
        ]


def get_flavor(client, name):
    flavor = None
    flavors = client.flavors.list()
    for f in flavors:
        if f.name == name:
            flavor = f
            break
    else:
        pytest.xfail("Failed to get flavor with name {}".format(name))
    return flavor


def get_image(client, name):
    images = list(client.images.list(filters={"name": name}))
    if not images:
        pytest.xfail("Failed to get image with name {}".format(name))
    return images[0]


def get_role(client, name=CONF.role_name):
    roles = client.roles.list(name=name)
    if not roles:
        pytest.xfail("Failed to get role with name {}".format(name))
    return roles[0]


def get_network(client, name):
    networks = client.list_networks(name=name).get("networks")
    if not networks:
        pytest.xfail("Failed to get network with name {}".format(name))
    return networks[0]


def get_volume_type(client, name=None):
    if name:
        volume_types = client.volume_types.list(search_opts={"name": name})
    else:
        volume_types = client.volume_types.list()
    if not volume_types:
        pytest.xfail("Failed to get volume type with name {}".format(name))
    return volume_types[0]
