<?php 
namespace core;

class Session{
    protected const KEY = "flash_messages";

    public function __construct(){
        if (\session_status() !== PHP_SESSION_ACTIVE) 
            session_start();
        $flashMessages = $_SESSION[self::KEY] ?? [];
        foreach ($flashMessages as &$flashMessage)
            $flashMessage['remove'] = true;
        $_SESSION[self::KEY] = $flashMessages;
    }

    public function setFlash($key, $message){
        $_SESSION[self::KEY][$key] = [
            'remove' => false,
            'value' => $message
        ];
    }

    public function getFlash($key){
        return $_SESSION[self::KEY][$key]['value'] ?? Null;
    }

    public function __destruct(){
        $flashMessages = $_SESSION[self::KEY] ?? [];

        foreach ($flashMessages as $key => $flashMessage){
            if ($flashMessage['remove'])
                unset($flashMessages[$key]);
        }

        $_SESSION[self::KEY] = $flashMessages;
    }

}