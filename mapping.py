import numpy as np

divice2mac = {
    "ds-ab": {
        "OppleLightLamp": "EC:0B:AE:4E:D1:4B",
        "ChintSocket": "88:97:46:26:CD:2A",
        "ChuangMiCamera": "78:8B:2A:60:80:53",
        "TpLinkCamera": "F4:6D:2F:BB:6E:D5",
        "YeelinkLightLamp0": "B4:60:ED:05:DA:A6",
        "YeelinkLightLamp1": "B4:60:ED:05:F5:68",
        "MiAiSoundbox0": "64:64:4A:55:A3:A7",
        "MiAiSoundbox1": "64:64:4A:55:A0:70",
        "TmallGenie0": "84:44:AF:3B:60:9D",
        "TmallGenie1": "84:44:AF:3B:60:A1",
        "TpLinkSocket0": "F4:6D:2F:BA:49:D9",
        "TpLinkSocket1": "F4:6D:2F:BA:4B:B3",
        "iPhone11": "02:2C:37:53:1D:EB",
        "iPhone13Mini": "B2:4B:42:D5:32:7C",
        "iPhone13Pro": "9A:27:62:7A:CB:57",
        "iPhone14ProMax": "6A:39:F4:DB:AE:B7",
    },
}


class DeviceMapping:
    def __init__(self, pattern: str):
        self.mac2token = dict(
            zip(
                list(divice2mac.get(pattern, "invalid pattern").values()),
                list(
                    np.arange(
                        0,
                        len(divice2mac.get(pattern, "invalid pattern")),
                        dtype=np.int32,
                    )
                ),
            )
        )
        self.token2mac = {v: k for k, v in self.mac2token.items()}
        self.mac2divice = {
            v: k for k, v in divice2mac.get(pattern, "invalid pattern").items()
        }

    def get_Token(self, mac: str):
        return self.mac2token.get(mac, -1)

    def get_divice(self, token: int):
        return self.mac2divice.get(self.token2mac.get(token, "unknown"), "unknown")
