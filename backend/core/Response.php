<?php
namespace core;

use helpers\SecurityHelper;

class Response{
    public function setStatusCode(int $code){
        http_response_code($code);
    }

    public function setResponseHeaders(array $headers){
        foreach($headers as $header)
            header($header);
    }

    public function returnJson(mixed $data, int $status_code){
        $this->setStatusCode($status_code);
        $this->setResponseHeaders(["content-type: application/json"]);
        echo json_encode($data);
        exit(0);
    }

    public function returnJsonEncrypted(mixed $data, int $status_code){
        $this->setStatusCode($status_code);
        $this->setResponseHeaders(["content-type: application/json"]);
        $helper = new SecurityHelper();
        $data = $helper->encryptSessionData($data);
        echo json_encode($data);
        exit(0);
    }
}