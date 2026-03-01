import json
import os
import websocket
import time

class VTSConnector:
    def __init__(self, host="127.0.0.1", port=8001):
        self.host = host
        self.port = port
        self.token_file = ".vts_token"
        self.token = self._load_token()
        self.plugin_name = "Yuuna-Project"
        self.developer = "Nonx2"
        self.connected = False
        self.authenticated = False
        self.hotkeys = []

    def _load_token(self):
        if os.path.exists(self.token_file):
            try:
                with open(self.token_file, "r") as f:
                    return f.read().strip()
            except:
                pass
        return None

    def _save_token(self, token):
        self.token = token
        try:
            with open(self.token_file, "w") as f:
                f.write(token)
        except:
            pass

    def _execute(self, request_type, data=None):
        """
        Executes a request by opening a new connection, authenticating if needed, 
        sending the request, and closing the connection.
        This is slow but extremely robust against connection drops.
        """
        ws = None
        try:
            url = f"ws://{self.host}:{self.port}"
            ws = websocket.create_connection(url, timeout=5)
            
            # --- Phase 1: Authentication ---
            if not self.token:
                # Request new token
                token_req = {
                    "apiName": "VTubeStudioPublicAPI",
                    "apiVersion": "1.0",
                    "requestID": "TokenReq",
                    "messageType": "AuthenticationTokenRequest",
                    "data": {
                        "pluginName": self.plugin_name,
                        "pluginDeveloper": self.developer
                    }
                }
                ws.send(json.dumps(token_req))
                res = json.loads(ws.recv())
                if res.get("messageType") == "AuthenticationTokenResponse":
                    self.token = res["data"]["authenticationToken"]
                    self._save_token(self.token)
                else:
                    return False, f"Token request failed: {res.get('data', {}).get('message', 'Unknown')}"

            # Authenticate with existing/new token
            auth_req = {
                "apiName": "VTubeStudioPublicAPI",
                "apiVersion": "1.0",
                "requestID": "AuthReq",
                "messageType": "AuthenticationRequest",
                "data": {
                    "pluginName": self.plugin_name,
                    "pluginDeveloper": self.developer,
                    "authenticationToken": self.token
                }
            }
            ws.send(json.dumps(auth_req))
            res = json.loads(ws.recv())
            
            if res.get("messageType") == "APIError" and res["data"].get("errorID") == 8:
                # Invalid token, clear and retry once
                self.token = None
                if os.path.exists(self.token_file):
                    os.remove(self.token_file)
                ws.close()
                return self._execute(request_type, data)

            if not (res.get("messageType") == "AuthenticationResponse" and res["data"]["authenticated"]):
                return False, f"Auth failed: {res.get('data', {}).get('message', 'Not authenticated')}"

            # --- Phase 2: Actual Request ---
            if request_type == "authenticate_only":
                return True, "Authenticated"

            msg = {
                "apiName": "VTubeStudioPublicAPI",
                "apiVersion": "1.0",
                "requestID": f"Req_{int(time.time())}",
                "messageType": request_type,
                "data": data if data else {}
            }
            ws.send(json.dumps(msg))
            res = json.loads(ws.recv())
            return True, res

        except Exception as e:
            return False, f"VTS Error: {str(e)}"
        finally:
            if ws:
                try:
                    ws.close()
                except:
                    pass

    def authenticate(self):
        success, res = self._execute("authenticate_only")
        if success:
            self.connected = True
            self.authenticated = True
            return True, "Authenticated"
        else:
            self.connected = False
            self.authenticated = False
            return False, res

    def get_hotkeys(self):
        success, res = self._execute("HotkeysInCurrentModelRequest")
        if success and res.get("messageType") == "HotkeysInCurrentModelResponse":
            self.hotkeys = res["data"]["availableHotkeys"]
            return self.hotkeys
        return []

    def trigger_hotkey(self, hotkey_id):
        success, res = self._execute("HotkeyTriggerRequest", {"hotkeyID": hotkey_id})
        if success and res.get("messageType") == "HotkeyTriggerResponse":
            return True, "Hotkey triggered"
        return False, res if not success else "Failed to trigger"

    def clear_token(self):
        self.token = None
        if os.path.exists(self.token_file):
            try:
                os.remove(self.token_file)
            except:
                pass
        self.authenticated = False
        self.connected = False
        return True, "Token cleared"
