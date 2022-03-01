<?php

namespace controllers\embedded;

use core\Response;
use core\Request;
use helpers\Helper;
use helpers\SecurityHelper;
use middlewares\EncryptionLayerMiddleWare;

class ApiSessionKeysGeneratorController extends \core\Controller
{
    private SecurityHelper $security_helper;

    public function __construct(){
        $this->response = new Response();
        $this->request = new Request();
        $this->security_helper = new SecurityHelper();
        $this->setMiddleWare(new EncryptionLayerMiddleWare());
    }

    public function generateSessionKeys(){
        $status_code = 200;
        $message = [
            'encryption_type' => Helper::isLogged() ? "symmetric" : "asymmetric"
        ];
        $error_message = "";
        $error_code = 0;

        if ($this->request->isPost()){
            if (isset($_POST, $_POST['secret_key'],
                    $_POST['encrypted_test'],
                    $_POST['decrypted_test'])){

                $secret_key = hex2bin($_POST['secret_key']);
                $encrypted_text = $_POST['encrypted_test'];
                $decrypted_text = $_POST['decrypted_test'];
                $decrypted_test = $this->security_helper->symmetricDecryption($encrypted_text, $secret_key);
                if ($decrypted_test == $decrypted_text) {
                    $_SESSION['secret_key'] = hex2bin($_POST['secret_key']);
                    $message['status'] = "Your secret key allocated.";
                }else{
                    $message['status'] = "you entered invalid secret key -> couldn't decrypt the data";
                    $error_message = "invalid secret key given";
                    $error_code = 0x2;
                }

            }else{
                $message['status'] = "Please enter a secret key to be allocated.";
                $error_message = "please enter a valid secret key to be allocated.";
                $error_code = 0x22;
            }
        }

        if (!isset($_SESSION['private_key'], $_SESSION['public_key']))
            $this->security_helper->generateKeyPairs($_SESSION['private_key'], $_SESSION['public_key']);
        $message['public_key'] = $_SESSION['public_key'];
        $response = Helper::generateApiResponse($message, $status_code, $error_message, $error_code);
        $this->response->returnJson($response, $status_code);
    }
}