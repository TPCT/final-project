<?php
namespace core;

class Redirection{
    public function redirect(string $route, bool $destroy_session = False, int $code=302){
        if ($destroy_session) session_destroy();
        header("Location: {$route}", true, $code);
        exit(0);
    }
}